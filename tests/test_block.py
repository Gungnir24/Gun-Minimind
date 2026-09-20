"""MiniMindBlock 单元测试: Pre-Norm 残差块的组装与接线

运行: python tests/test_block.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from model.model_minimind import (
    MiniMindConfig, MiniMindBlock, Attention, FeedForward, MOEFeedForward, RMSNorm,
    precompute_freqs_cis,
)

HIDDEN, N_HEADS, N_KV, HEAD_DIM = 64, 4, 2, 16


def make_block(seed=0, moe=False):
    torch.manual_seed(seed)
    cfg = MiniMindConfig(
        hidden_size=HIDDEN, num_hidden_layers=2,
        num_attention_heads=N_HEADS, num_key_value_heads=N_KV,
        head_dim=HEAD_DIM, dropout=0.0, flash_attn=False, use_moe=moe,
    )
    block = MiniMindBlock(0, cfg)
    block.eval()
    return block


def make_pos(end=64):
    return precompute_freqs_cis(HEAD_DIM, end, rope_base=1e6)


def test_output_shape():
    """输出 [b, s, hidden]; use_cache=False 时不返回缓存"""
    block = make_block()
    cos, sin = make_pos()
    x = torch.randn(2, 5, HIDDEN)
    out, past = block(x, (cos[:5], sin[:5]))
    assert out.shape == (2, 5, HIDDEN), f"输出形状错误: {tuple(out.shape)}"
    assert past is None, "use_cache=False 时 past 应该是 None"


def test_structure():
    """四件套: Attention + 两个 RMSNorm + (MoE)FeedForward"""
    block = make_block()
    assert isinstance(block.self_attn, Attention)
    assert isinstance(block.input_layernorm, RMSNorm), "进注意力前的 norm 应该是 RMSNorm"
    assert isinstance(block.post_attention_layernorm, RMSNorm), "进 FFN 前的 norm 應該是 RMSNorm"
    assert isinstance(block.mlp, FeedForward)
    assert block.input_layernorm.weight.shape == (HIDDEN,)
    assert block.post_attention_layernorm.weight.shape == (HIDDEN,)


def test_moe_swap():
    """use_moe=True 时 mlp 换成 MOEFeedForward, 整块照常工作"""
    block = make_block(moe=True)
    assert isinstance(block.mlp, MOEFeedForward), "use_moe=True 时 mlp 应该是 MOEFeedForward"
    cos, sin = make_pos()
    x = torch.randn(1, 4, HIDDEN)
    out, _ = block(x, (cos[:4], sin[:4]))
    assert out.shape == (1, 4, HIDDEN)


def test_matches_manual_wiring():
    """接线对拍: out == x + Attn(N1(x)) + MLP(N2(x + Attn(N1(x))))"""
    block = make_block(seed=42)
    cos, sin = make_pos()
    x = torch.randn(2, 5, HIDDEN)
    out, _ = block(x, (cos[:5], sin[:5]))
    h_attn, _ = block.self_attn(block.input_layernorm(x), (cos[:5], sin[:5]))
    expected = x + h_attn
    expected = expected + block.mlp(block.post_attention_layernorm(expected))
    assert torch.allclose(out, expected, atol=1e-5), \
        "与 Pre-Norm 残差手工接线不一致: 检查 norm 的位置(先 norm 再进子层)和两次残差相加"


def test_residual_bypass():
    """残差是真正的旁路: 两个子层输出清零后, 块输出精确等于输入"""
    block = make_block(seed=1)
    with torch.no_grad():
        block.self_attn.o_proj.weight.zero_()
        block.mlp.down_proj.weight.zero_()
    cos, sin = make_pos()
    x = torch.randn(2, 5, HIDDEN)
    out, _ = block(x, (cos[:5], sin[:5]))
    assert torch.equal(out, x), \
        "两个子层都输出 0 时, 块输出应该原样等于输入 —— 残差流是不被改写的旁路"


def test_kv_cache_incremental():
    """整块也支持 KV Cache: 逐 token 增量 == 一次全量"""
    block = make_block(seed=2)
    cos, sin = make_pos()
    torch.manual_seed(3)
    x = torch.randn(1, 6, HIDDEN)
    full_out, _ = block(x, (cos[:6], sin[:6]))

    past = None
    outs = []
    for t in range(6):
        out_t, past = block(x[:, t:t + 1], (cos[t:t + 1], sin[t:t + 1]),
                            past_key_value=past, use_cache=True)
        outs.append(out_t)
        assert past[0].shape == (1, t + 1, N_KV, HEAD_DIM), \
            f"缓存形状错误: {tuple(past[0].shape)}, 应为 (1, t+1, {N_KV}, {HEAD_DIM})"
    inc_out = torch.cat(outs, dim=1)
    assert torch.allclose(full_out, inc_out, atol=1e-4), "块级增量与全量不一致"


def test_gradients_flow():
    """所有参数都有非零梯度; 恰好 11 个参数张量(4 qkvo + 2 qk_norm + 2 block norm + 3 mlp)"""
    block = make_block(seed=4)
    cos, sin = make_pos()
    x = torch.randn(2, 3, HIDDEN)
    out, _ = block(x, (cos[:3], sin[:3]))
    out.sum().backward()
    n = 0
    for name, p in block.named_parameters():
        assert p.grad is not None, f"{name} 没有梯度"
        assert p.grad.abs().sum() > 0, f"{name} 梯度全零"
        n += 1
    assert n == 11, f"参数张量数应该是 11(qkvo 4 + qk/k_norm 2 + 块内 norm 2 + mlp 3), 实际 {n}"


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
