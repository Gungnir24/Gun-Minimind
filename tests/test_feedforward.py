"""FeedForward (SwiGLU) 单元测试

运行: python tests/test_feedforward.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from model.model_minimind import MiniMindConfig, FeedForward

HIDDEN = 64
INTERMEDIATE = 256  # ceil(64 * pi / 64) * 64


def make_config(dropout=0.0):
    return MiniMindConfig(hidden_size=HIDDEN, num_hidden_layers=2, dropout=dropout)


def make_ffn(seed=0, dropout=0.0):
    torch.manual_seed(seed)
    ffn = FeedForward(make_config(dropout))
    ffn.eval()
    return ffn


def test_output_shape():
    """输出形状 [b, s, hidden]"""
    ffn = make_ffn()
    x = torch.randn(2, 5, HIDDEN)
    out = ffn(x)
    assert out.shape == (2, 5, HIDDEN), f"输出形状错误: {tuple(out.shape)}"


def test_projection_shapes():
    """gate/up: hidden -> intermediate; down: intermediate -> hidden; 全部无 bias"""
    ffn = make_ffn()
    assert ffn.gate_proj.weight.shape == (INTERMEDIATE, HIDDEN), \
        f"gate_proj 形状错误: {tuple(ffn.gate_proj.weight.shape)}, 应为 (256, 64)"
    assert ffn.up_proj.weight.shape == (INTERMEDIATE, HIDDEN)
    assert ffn.down_proj.weight.shape == (HIDDEN, INTERMEDIATE), \
        "down_proj 应该从 intermediate 收回 hidden"
    assert ffn.gate_proj.bias is None, "投影不应该有 bias"


def test_intermediate_size_formula():
    """config 默认 intermediate = ceil(hidden * pi / 64) * 64: 64 -> 256, 768 -> 2432"""
    c1 = MiniMindConfig(hidden_size=64)
    assert c1.intermediate_size == 256, \
        f"hidden=64 时 intermediate 应为 256, 实际 {c1.intermediate_size}"
    c2 = MiniMindConfig(hidden_size=768)
    assert c2.intermediate_size == 2432, \
        f"hidden=768 时 intermediate 应为 2432, 实际 {c2.intermediate_size}"


def test_matches_manual_formula():
    """与手算公式一致: down( silu(gate(x)) * up(x) ), 其中 silu(t) = t * sigmoid(t)"""
    ffn = make_ffn(seed=42)
    x = torch.randn(2, 5, HIDDEN)
    out = ffn(x)
    gate = x @ ffn.gate_proj.weight.T
    up = x @ ffn.up_proj.weight.T
    silu = gate * torch.sigmoid(gate)
    expected = (silu * up) @ ffn.down_proj.weight.T
    assert torch.allclose(out, expected, atol=1e-5), \
        "与 SwiGLU 手算公式不一致: 检查是否漏了 silu、乘法顺序、down_proj 位置"


def test_gate_controls_output():
    """乘法门控: gate 支路权重清零 -> silu(0)=0 -> 输出全 0(证明是相乘不是相加)"""
    ffn = make_ffn(seed=1)
    x = torch.randn(1, 3, HIDDEN)
    with torch.no_grad():
        ffn.gate_proj.weight.zero_()
    out = ffn(x)
    assert torch.all(out == 0), "gate 支路被关死时输出应该全 0: 检查两条支路是不是逐元素相乘"


def test_per_token_independence():
    """FFN 逐位置独立: 改一个 token 不影响其他任何 token(和 attention 形成对比)"""
    ffn = make_ffn(seed=2)
    x = torch.randn(1, 4, HIDDEN)
    out1 = ffn(x)
    x2 = x.clone()
    x2[:, 2] = torch.randn(HIDDEN) * 10
    out2 = ffn(x2)
    assert torch.allclose(out1[:, :2], out2[:, :2], atol=1e-6), \
        "改 token 2 不该影响 token 0/1: FFN 不应该有跨位置混合"
    assert torch.allclose(out1[:, 3:], out2[:, 3:], atol=1e-6), \
        "改 token 2 不该影响 token 3"
    assert not torch.allclose(out1[:, 2], out2[:, 2]), "被改的 token 自身输出应该变"


def test_gradients_flow():
    """三个投影的梯度都存在且非零"""
    ffn = make_ffn(seed=3)
    x = torch.randn(2, 3, HIDDEN)
    ffn(x).sum().backward()
    for name in ["gate_proj", "up_proj", "down_proj"]:
        g = getattr(ffn, name).weight.grad
        assert g is not None, f"{name} 没有梯度"
        assert g.abs().sum() > 0, f"{name} 梯度全零"


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
