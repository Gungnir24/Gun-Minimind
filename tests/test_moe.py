"""MOEFeedForward 单元测试: 路由 + 派发 + 负载均衡

运行: python tests/test_moe.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn.functional as F

from model.model_minimind import MiniMindConfig, MOEFeedForward, FeedForward

HIDDEN = 64
N_EXPERTS = 4
INTERMEDIATE = 256


def make_config(topk=1):
    return MiniMindConfig(
        hidden_size=HIDDEN, num_hidden_layers=2,
        num_experts=N_EXPERTS, num_experts_per_tok=topk,
    )


def make_moe(topk=1, seed=0):
    torch.manual_seed(seed)
    moe = MOEFeedForward(make_config(topk))
    moe.eval()
    return moe


def manual_forward(moe, x, topk):
    """按算法手工重算一遍路由+派发(k=topk), 用于对拍"""
    b, s, h = x.shape
    x_flat = x.view(-1, h)
    scores = F.softmax(moe.gate(x_flat), dim=-1)
    topk_w, topk_i = torch.topk(scores, k=topk, dim=-1, sorted=False)
    topk_w = topk_w / (topk_w.sum(dim=-1, keepdim=True) + 1e-20)
    y = torch.zeros_like(x_flat)
    for i, expert in enumerate(moe.experts):
        mask = (topk_i == i)
        if mask.any():
            token_idx = mask.any(dim=-1).nonzero().flatten()
            weight = topk_w[mask].view(-1, 1)
            y.index_add_(0, token_idx, expert(x_flat[token_idx]) * weight)
    return y.view(b, s, h)


def test_output_shape():
    """输出形状 [b, s, hidden]"""
    moe = make_moe()
    x = torch.randn(2, 5, HIDDEN)
    assert moe(x).shape == (2, 5, HIDDEN), f"输出形状错误: {tuple(moe(x).shape)}"


def test_structure():
    """gate: hidden -> num_experts 无 bias; experts: N 个 FeedForward"""
    moe = make_moe()
    assert moe.gate.weight.shape == (N_EXPERTS, HIDDEN), \
        f"gate 形状错误: {tuple(moe.gate.weight.shape)}, 应为 (4, 64) —— 每个专家一个分数"
    assert moe.gate.bias is None
    assert len(moe.experts) == N_EXPERTS
    for e in moe.experts:
        assert isinstance(e, FeedForward), "专家应该是 FeedForward 实例"
        assert e.gate_proj.weight.shape == (INTERMEDIATE, HIDDEN)


def test_matches_manual_k1():
    """k=1: 整条路由+派发流程与手工重算一致"""
    moe = make_moe(seed=42)
    x = torch.randn(2, 5, HIDDEN)
    expected = manual_forward(moe, x, topk=1)
    assert torch.allclose(moe(x), expected, atol=1e-5), \
        "与手工重算的 top-1 路由结果不一致: 检查 topk、归一化、index_add_ 的行号对齐"


def test_matches_manual_k2():
    """k=2: 两个专家加权混合, 仍然与手工重算一致"""
    moe = make_moe(topk=2, seed=42)
    x = torch.randn(2, 5, HIDDEN)
    expected = manual_forward(moe, x, topk=2)
    assert torch.allclose(moe(x), expected, atol=1e-5), \
        "与手工重算的 top-2 路由结果不一致: 检查一个 token 从两个专家收货时的权重与累加"


def test_all_to_one_expert():
    """gate 改成只给专家 0 打高分 -> 所有 token 走专家 0, 输出 == experts[0](x)"""
    moe = make_moe(seed=1)
    x = torch.randn(1, 6, HIDDEN)
    with torch.no_grad():
        moe.gate.weight.zero_()
        moe.gate.weight[0, 0] = 1.0   # 专家 0 的 logit = x[..., 0]
    x[..., 0] = 10.0                  # 所有 token 的 logit_0 = 10 > 其余的 0, 专家 0 全胜
    out = moe(x)
    expected = moe.experts[0](x)
    assert torch.allclose(out, expected, atol=1e-6), \
        "全部 token 路由到专家 0 时, 输出应该就是专家 0 的输出(k=1 归一化后权重恰为 1)"


def test_per_token_independence():
    """MoE 同样逐位置独立: 路由决策只看 token 自己"""
    moe = make_moe(seed=2)
    x = torch.randn(1, 4, HIDDEN)
    out1 = moe(x)
    x2 = x.clone()
    x2[:, 2] = torch.randn(HIDDEN) * 10
    out2 = moe(x2)
    assert torch.allclose(out1[:, :2], out2[:, :2], atol=1e-6), "改 token 2 不该影响 token 0/1"
    assert torch.allclose(out1[:, 3:], out2[:, 3:], atol=1e-6), "改 token 2 不该影响 token 3"
    assert not torch.allclose(out1[:, 2], out2[:, 2]), "被改的 token 自身输出应该变"


def test_aux_loss_matches_formula():
    """aux_loss = N * sum(load_i * mean_score_i) * coef; eval 模式下为 0"""
    moe = make_moe(seed=3)
    moe.train()
    x = torch.randn(2, 5, HIDDEN)
    moe(x)
    scores = F.softmax(moe.gate(x.view(-1, HIDDEN)), dim=-1)
    _, topk_i = torch.topk(scores, k=1, dim=-1, sorted=False)
    load = F.one_hot(topk_i, N_EXPERTS).float().mean(0)
    coef = moe.config.router_aux_loss_coef
    expected = (load * scores.mean(0)).sum() * N_EXPERTS * coef
    assert torch.allclose(moe.aux_loss, expected, rtol=1e-4), "aux_loss 公式对不上"
    moe.eval()
    moe(x)
    assert float(moe.aux_loss) == 0.0, "eval 模式下 aux_loss 应该是 0"


def test_aux_loss_penalizes_skew():
    """全挤一个专家的 aux_loss 明显大于四专家均衡负载(均衡是 aux 的最小点)"""
    moe = make_moe(seed=4)
    moe.train()
    with torch.no_grad():
        moe.gate.weight.zero_()
        moe.gate.weight[0, 0] = 1.0
    x_skew = torch.randn(1, 4, HIDDEN)
    x_skew[..., 0] = 10.0                # 4 个 token 全部 -> 专家 0
    moe(x_skew)
    aux_skew = float(moe.aux_loss)

    with torch.no_grad():
        moe.gate.weight.zero_()
        for i in range(N_EXPERTS):
            moe.gate.weight[i, i] = 1.0   # 专家 i 只认特征 i
    x_bal = torch.zeros(1, N_EXPERTS, HIDDEN)
    for t in range(N_EXPERTS):
        x_bal[0, t, t] = 10.0             # token t -> 专家 t, 完美均衡
    moe(x_bal)
    aux_bal = float(moe.aux_loss)

    assert aux_skew > aux_bal * 2, \
        f"挤在一个专家(aux={aux_skew:.6f})应该比均衡负载(aux={aux_bal:.6f})罚得多"


def test_idle_expert_still_has_grad():
    """训练模式下, 没分到 token 的专家梯度也必须"存在"(DDP 兼容的幽灵梯度)"""
    moe = make_moe(seed=5)
    moe.train()
    with torch.no_grad():
        moe.gate.weight.zero_()
        moe.gate.weight[0, 0] = 1.0
    x = torch.randn(1, 3, HIDDEN)
    x[..., 0] = 10.0
    moe(x).sum().backward()
    for i in range(1, N_EXPERTS):
        g = moe.experts[i].gate_proj.weight.grad
        assert g is not None, \
            f"专家 {i} 这批没接到 token, 但梯度必须存在(可为全 0), 否则 DDP 多卡训练会卡死"
    assert moe.experts[0].gate_proj.weight.grad.abs().sum() > 0, "接单的专家 0 应该有真实梯度"


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for name, t in tests:
        try:
            t()
            print(f"  [PASS] {name}")
            passed += 1
        except NotImplementedError as e:
            print(f"  [FAIL] {name}: {e}")
        except AssertionError as e:
            print(f"  [FAIL] {name}: {e}")
        except Exception as e:
            print(f"  [FAIL] {name}: {type(e).__name__}: {e}")
    print(f"\n  {passed}/{len(tests)} tests passed")
    sys.exit(0 if passed == len(tests) else 1)
