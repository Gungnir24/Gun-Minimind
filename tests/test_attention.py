"""Attention 单元测试: GQA + QK-Norm + KV Cache + causal mask

运行: python tests/test_attention.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from model.model_minimind import MiniMindConfig, Attention, repeat_kv, precompute_freqs_cis

HIDDEN, N_HEADS, N_KV, HEAD_DIM = 64, 4, 2, 16


def make_config(flash=False):
    return MiniMindConfig(
        hidden_size=HIDDEN, num_hidden_layers=2, num_attention_heads=N_HEADS,
        num_key_value_heads=N_KV, head_dim=HEAD_DIM, dropout=0.0, flash_attn=flash,
    )


def make_attn(flash=False, seed=0):
    torch.manual_seed(seed)
    attn = Attention(make_config(flash))
    attn.eval()
    return attn


def make_pos(end=64):
    return precompute_freqs_cis(HEAD_DIM, end, rope_base=1e6)


def test_output_shape():
    """输出 [b, s, hidden]; use_cache=False 时不返回缓存"""
    attn = make_attn()
    cos, sin = make_pos()
    x = torch.randn(2, 5, HIDDEN)
    out, past_kv = attn(x, (cos[:5], sin[:5]))
    assert out.shape == (2, 5, HIDDEN), f"输出形状错误: {tuple(out.shape)}"
    assert past_kv is None, "use_cache=False 时 past_kv 应该是 None"


def test_projection_shapes():
    """GQA 结构: k/v 投影输出只有 q 的一半; 所有投影无 bias"""
    attn = make_attn()
    assert attn.q_proj.weight.shape == (N_HEADS * HEAD_DIM, HIDDEN)
    assert attn.k_proj.weight.shape == (N_KV * HEAD_DIM, HIDDEN), "k_proj 输出应该是 KV头数×head_dim(减半)"
    assert attn.v_proj.weight.shape == (N_KV * HEAD_DIM, HIDDEN)
    assert attn.o_proj.weight.shape == (HIDDEN, N_HEADS * HEAD_DIM)
    assert attn.q_proj.bias is None, "投影不应该有 bias"


def test_repeat_kv():
    """KV 头复制语义: Q 头 0,1 共享 KV 头 0; Q 头 2,3 共享 KV 头 1"""
    x = torch.arange(12).float().reshape(1, 3, 2, 2)  # [b=1, s=3, kv=2, d=2]
    out = repeat_kv(x, 2)
    assert out.shape == (1, 3, 4, 2), f"形状错误: {tuple(out.shape)}"
    assert torch.equal(out[:, :, 0], x[:, :, 0]) and torch.equal(out[:, :, 1], x[:, :, 0]), \
        "扩出来的第 0、1 个头都应该来自 KV 头 0"
    assert torch.equal(out[:, :, 2], x[:, :, 1]) and torch.equal(out[:, :, 3], x[:, :, 1]), \
        "扩出来的第 2、3 个头都应该来自 KV 头 1"
    assert torch.equal(repeat_kv(x, 1), x), "n_rep=1 时应该原样返回"


def test_causality():
    """因果性: 改变未来 token, 不影响之前 token 的输出"""
    attn = make_attn()
    cos, sin = make_pos()
    torch.manual_seed(1)
    x = torch.randn(1, 6, HIDDEN)
    out1, _ = attn(x, (cos[:6], sin[:6]))
    x2 = x.clone()
    x2[:, -1] = torch.randn(HIDDEN) * 10  # 大幅改动最后一个 token
    out2, _ = attn(x2, (cos[:6], sin[:6]))
    assert torch.allclose(out1[:, :5], out2[:, :5], atol=1e-5), "改动未来 token 影响了过去的输出: causal mask 有问题"
    assert not torch.allclose(out1[:, 5], out2[:, 5]), "被改动的 token 自身的输出竟然没变"


def test_padding_mask():
    """padding mask: 遮住位置 2 后, 位置 0/1 不受影响, 位置 2 之后全部改变"""
    attn = make_attn()
    cos, sin = make_pos()
    torch.manual_seed(3)
    x = torch.randn(1, 5, HIDDEN)
    out1, _ = attn(x, (cos[:5], sin[:5]), attention_mask=None)
    mask = torch.ones(1, 5)
    mask[:, 2] = 0  # 遮住位置 2 这个 key
    out2, _ = attn(x, (cos[:5], sin[:5]), attention_mask=mask)
    assert torch.allclose(out1[:, :2], out2[:, :2], atol=1e-4), "遮住位置2不该影响位置0,1(它们本来就看不到位置2)"
    assert not torch.allclose(out1[:, 2:], out2[:, 2:]), "遮住位置2应该改变位置2及以后的输出"


def test_kv_cache_incremental():
    """KV Cache 核心: 逐 token 增量前向 == 一次全量前向"""
    attn = make_attn()
    cos, sin = make_pos()
    torch.manual_seed(2)
    x = torch.randn(1, 6, HIDDEN)
    full_out, _ = attn(x, (cos[:6], sin[:6]))  # 全量一次算完

    past = None
    outs = []
    for t in range(6):
        out_t, past = attn(x[:, t:t + 1], (cos[t:t + 1], sin[t:t + 1]), past_key_value=past, use_cache=True)
        outs.append(out_t)
        assert past[0].shape == (1, t + 1, N_KV, HEAD_DIM), \
            f"缓存形状错误: {tuple(past[0].shape)}, 应为 (b, t+1, kv头, head_dim) —— 存未 repeat 的 KV"
    inc_out = torch.cat(outs, dim=1)
    diff = (full_out - inc_out).abs().max().item()
    assert torch.allclose(full_out, inc_out, atol=1e-4), \
        f"增量与全量不一致 (max diff: {diff:.2e}): 检查 cache 的 cat 方向、RoPE 位置切片、causal mask"


def test_flash_matches_manual():
    """flash(SDPA) 路径与手动路径输出一致(先实现手动路径, 最后加 flash 再跑这个)"""
    attn_flash = make_attn(flash=True, seed=0)
    attn_manual = make_attn(flash=False, seed=0)
    cos, sin = make_pos()
    torch.manual_seed(4)
    x = torch.randn(2, 7, HIDDEN)
    out_f, _ = attn_flash(x, (cos[:7], sin[:7]))
    out_m, _ = attn_manual(x, (cos[:7], sin[:7]))
    diff = (out_f - out_m).abs().max().item()
    assert torch.allclose(out_f, out_m, atol=1e-4), f"flash 与手动路径不一致 (max diff: {diff:.2e})"


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
