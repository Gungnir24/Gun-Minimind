"""RMSNorm 单元测试

运行: python tests/test_rmsnorm.py
全部通过 = 你的实现正确
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from model.model_minimind import RMSNorm


def test_output_shape():
    """输出形状必须与输入一致: [batch, seq, dim] -> [batch, seq, dim]"""
    x = torch.randn(2, 16, 768)
    out = RMSNorm(768)(x)
    assert out.shape == x.shape, f"shape 不一致: {tuple(out.shape)} vs {tuple(x.shape)}"


def test_unit_rms():
    """weight 全 1 时, 每个 token 输出的 RMS 应该接近 1"""
    x = torch.randn(4, 32, 768)
    out = RMSNorm(768)(x)
    rms = out.pow(2).mean(-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-3), \
        f"输出 RMS 不为 1: 平均 {rms.mean().item():.4f}"


def test_matches_manual_formula():
    """与手写公式 y = x / sqrt(mean(x^2)+eps) * weight 逐元素一致"""
    torch.manual_seed(42)
    x = torch.randn(2, 8, 64)
    norm = RMSNorm(64, eps=1e-6)
    with torch.no_grad():
        norm.weight.copy_(torch.randn(64) * 0.5 + 1.0)
    out = norm(x)
    rms = x.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt()
    expected = x / rms * norm.weight
    assert torch.allclose(out, expected, atol=1e-4), "与手写公式偏差过大"


def test_matches_torch_builtin():
    """与 PyTorch 官方 torch.nn.RMSNorm 交叉验证 (torch>=2.4 才有)"""
    torch.manual_seed(0)
    x = torch.randn(2, 8, 64)
    mine = RMSNorm(64, eps=1e-6)
    official = torch.nn.RMSNorm(64, eps=1e-6)
    assert torch.allclose(mine(x), official(x), atol=1e-5), "与官方实现偏差过大"


def test_fp16_numerical_stability():
    """fp16 输入: 统计量若在 fp16 下计算, 平方和会溢出为 inf, 输出 RMS 会归零"""
    x = torch.randn(2, 8, 768).half() * 300  # 元素约±300, 平方约 9e4, 已超 fp16 上限 65504
    out = RMSNorm(768)(x)
    assert torch.isfinite(out).all(), "输出出现 inf/nan"
    rms = out.float().pow(2).mean(-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-2), \
        "fp16 输出 RMS 不为 1: 统计量是否在 float32 上计算?"


def test_not_layernorm():
    """RMSNorm 不做均值中心化: 给输入加一个整体偏移, 输出均值不应归零"""
    x = torch.randn(2, 8, 64) + 5.0  # 均值约 5 的输入
    out = RMSNorm(64)(x)
    assert (out.mean(-1).abs() > 0.1).all(), \
        "输出均值约等于 0: 你可能多做了均值中心化(那是 LayerNorm 的行为)"


def test_gradients_flow():
    """梯度必须能回传到 weight 和输入"""
    norm = RMSNorm(768)
    x = torch.randn(2, 8, 768, requires_grad=True)
    norm(x).sum().backward()
    assert norm.weight.grad is not None and torch.isfinite(norm.weight.grad).all(), \
        "weight 没有收到梯度: weight 是否用了 nn.Parameter?"
    assert x.grad is not None and torch.isfinite(x.grad).all(), "输入没有收到梯度"


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
