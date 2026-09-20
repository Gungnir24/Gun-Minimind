"""RoPE 旋转位置编码单元测试

运行: python tests/test_rope.py
全部通过 = 你的实现正确
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from model.model_minimind import precompute_freqs_cis, apply_rotary_pos_emb

D = 64  # head_dim


def test_table_shape():
    """表形状 [end, dim]; 位置 0 的 cos 全 1, sin 全 0(不旋转)"""
    cos, sin = precompute_freqs_cis(D, 128, rope_base=1e6)
    assert cos.shape == (128, D), f"cos 形状错误: {tuple(cos.shape)}"
    assert sin.shape == (128, D), f"sin 形状错误: {tuple(sin.shape)}"
    assert torch.allclose(cos[0], torch.ones(D), atol=1e-6), "位置 0 的 cos 应该全为 1"
    assert torch.allclose(sin[0], torch.zeros(D), atol=1e-6), "位置 0 的 sin 应该全为 0"


def test_position_zero_is_identity():
    """位置 0 不旋转: q/k 原样返回"""
    q = torch.randn(2, 1, 8, D)
    k = torch.randn(2, 1, 8, D)
    cos, sin = precompute_freqs_cis(D, 4)
    qe, ke = apply_rotary_pos_emb(q, k, cos[0:1], sin[0:1])
    assert torch.allclose(q, qe, atol=1e-5), "位置 0 的 q 被改变了"
    assert torch.allclose(k, ke, atol=1e-5), "位置 0 的 k 被改变了"


def test_rotation_changes_values():
    """位置 > 0 时向量确实被旋转(和输入不同)"""
    q = torch.randn(1, 1, 4, D)
    cos, sin = precompute_freqs_cis(D, 4)
    qe, _ = apply_rotary_pos_emb(q, torch.zeros_like(q), cos[1:2], sin[1:2])
    assert not torch.allclose(q, qe), "位置 1 的输出和输入完全一样: 旋转没生效"


def test_norm_preserved():
    """旋转是正交变换: 每个 token 向量的范数不变"""
    q = torch.randn(2, 5, 8, D)
    k = torch.randn(2, 5, 8, D)
    cos, sin = precompute_freqs_cis(D, 8)
    qe, ke = apply_rotary_pos_emb(q, k, cos[2:7], sin[2:7])  # 从位置 2 开始的 5 个位置
    assert torch.allclose(q.norm(dim=-1), qe.norm(dim=-1), atol=1e-3), "旋转改变了 q 的范数"
    assert torch.allclose(k.norm(dim=-1), ke.norm(dim=-1), atol=1e-3), "旋转改变了 k 的范数"


def test_relative_position_invariance():
    """RoPE 的核心性质: 旋转后的内积只依赖相对位置 m-n, 整体平移不变"""
    torch.manual_seed(0)
    q = torch.randn(1, 1, 4, D)
    k = torch.randn(1, 1, 4, D)
    cos, sin = precompute_freqs_cis(D, 128)
    m, n, s = 10, 3, 20
    q_m, _ = apply_rotary_pos_emb(q, k, cos[m:m + 1], sin[m:m + 1])          # q 转到位置 m
    _, k_n = apply_rotary_pos_emb(q, k, cos[n:n + 1], sin[n:n + 1])         # k 转到位置 n
    q_ms, _ = apply_rotary_pos_emb(q, k, cos[m + s:m + s + 1], sin[m + s:m + s + 1])
    _, k_ns = apply_rotary_pos_emb(q, k, cos[n + s:n + s + 1], sin[n + s:n + s + 1])
    score = (q_m * k_n).sum(-1)
    score_shifted = (q_ms * k_ns).sum(-1)
    assert torch.allclose(score, score_shifted, atol=1e-4), \
        "两个位置同时平移后内积变了: 相对位置性质被破坏, 检查 cos/sin 角度公式"


def test_matches_pair_rotation_formula():
    """与 2D 逐对旋转公式等价: 第 i 维与第 i+dim/2 维组成一对, 按角度 t*theta_i 旋转"""
    torch.manual_seed(42)
    base = 10000.0
    cos, sin = precompute_freqs_cis(D, 8, rope_base=base)
    q = torch.randn(2, 1, 4, D)
    t = 3
    q_rot, _ = apply_rotary_pos_emb(q, torch.zeros_like(q), cos[t:t + 1], sin[t:t + 1])
    # 手动公式: theta_j = base^(-2j/dim), 角度 = t * theta_j
    theta = base ** (-torch.arange(0, D, 2).float() / D)  # [D/2]
    angle = t * theta
    x_i, x_j = q[..., :D // 2], q[..., D // 2:]
    exp_i = x_i * angle.cos() - x_j * angle.sin()
    exp_j = x_j * angle.cos() + x_i * angle.sin()
    expected = torch.cat([exp_i, exp_j], dim=-1)
    assert torch.allclose(q_rot, expected, atol=1e-4), \
        "与逐对 2D 旋转公式不一致: 检查 theta 频率公式或 rotate_half 的对半劈开布局"


def test_dtype_preserved():
    """fp16 输入 -> fp16 输出(cos/sin 是 fp32, 算完必须转回)"""
    q = torch.randn(1, 2, 4, D).half()
    k = torch.randn(1, 2, 4, D).half()
    cos, sin = precompute_freqs_cis(D, 4)
    qe, ke = apply_rotary_pos_emb(q, k, cos[1:3], sin[1:3])
    assert qe.dtype == torch.float16, f"q_embed dtype 变成了 {qe.dtype}"
    assert ke.dtype == torch.float16, f"k_embed dtype 变成了 {ke.dtype}"


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
