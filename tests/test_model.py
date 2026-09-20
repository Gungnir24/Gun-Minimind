"""MiniMindModel 单元测试: 词嵌入 -> 层堆叠 -> 最终 Norm -> aux_loss 汇总

运行: python tests/test_model.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from model.model_minimind import (
    MiniMindConfig, MiniMindModel, MiniMindBlock, RMSNorm,
)

HIDDEN, N_HEADS, N_KV, HEAD_DIM = 64, 4, 2, 16
VOCAB, MAXPOS, LAYERS = 100, 64, 2


def make_model(seed=0, moe=False):
    torch.manual_seed(seed)
    cfg = MiniMindConfig(
        hidden_size=HIDDEN, num_hidden_layers=LAYERS, vocab_size=VOCAB,
        num_attention_heads=N_HEADS, num_key_value_heads=N_KV,
        head_dim=HEAD_DIM, dropout=0.0, flash_attn=False,
        max_position_embeddings=MAXPOS, use_moe=moe,
    )
    model = MiniMindModel(cfg)
    model.eval()
    return model


def rand_ids(b, s, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    return torch.randint(0, VOCAB, (b, s))


def test_output_shape():
    """[b,s] ids -> [b,s,hidden]; use_cache=False 时 presents 全 None, dense 的 aux 为 0"""
    model = make_model()
    ids = rand_ids(2, 5)
    hidden, presents, aux = model(ids)
    assert hidden.shape == (2, 5, HIDDEN), f"输出形状错误: {tuple(hidden.shape)}"
    assert presents == [None] * LAYERS, "use_cache=False 时 presents 应该是 [None] * 层数"
    assert float(aux) == 0.0, "dense 模型的 aux_loss 应该是 0"


def test_structure():
    """嵌入形状 / 层数 / 最终 norm / RoPE 表形状与取值 / 不进 state_dict"""
    model = make_model()
    assert model.embed_tokens.weight.shape == (VOCAB, HIDDEN)
    assert len(model.layers) == LAYERS
    assert all(isinstance(l, MiniMindBlock) for l in model.layers)
    assert isinstance(model.norm, RMSNorm), "收尾的最终 norm 应该是 RMSNorm"
    assert model.freqs_cos.shape == (MAXPOS, HEAD_DIM), \
        f"RoPE 表形状错误: {tuple(model.freqs_cos.shape)}, 应为 [max_position, head_dim]"
    assert torch.allclose(model.freqs_cos[0], torch.ones(HEAD_DIM), atol=1e-6), \
        "位置 0 的 cos 应该全 1"
    assert torch.allclose(model.freqs_sin[0], torch.zeros(HEAD_DIM), atol=1e-6), \
        "位置 0 的 sin 应该全 0"
    assert "freqs_cos" not in model.state_dict(), "RoPE 表不该进 state_dict(persistent=False)"
    assert "embed_tokens.weight" in model.state_dict(), "嵌入权重应该在 state_dict 里"


def test_matches_manual_wiring():
    """接线对拍: embed -> 逐层堆叠 -> 最终 norm"""
    model = make_model(seed=42)
    ids = rand_ids(2, 5)
    out, _, _ = model(ids)
    h = model.embed_tokens(ids)
    pos = (model.freqs_cos[:5], model.freqs_sin[:5])
    for layer in model.layers:
        h, _ = layer(h, pos)
    expected = model.norm(h)
    assert torch.allclose(out, expected, atol=1e-5), \
        "与手工接线不一致: 检查最终 norm 有没有加、各层是否用同一份 position_embeddings"


def test_incremental_equals_full():
    """整模型带缓存: 逐 token 增量 == 一次全量; 缓存是每层一个的 list"""
    model = make_model(seed=2)
    ids = rand_ids(1, 6, seed=3)
    full_out, _, _ = model(ids)

    past = None
    outs = []
    for t in range(6):
        out_t, presents, _ = model(ids[:, t:t + 1], past_key_values=past, use_cache=True)
        outs.append(out_t)
        past = presents
        assert len(presents) == LAYERS, "presents 应该每层一个缓存条目"
        for p in presents:
            assert p[0].shape == (1, t + 1, N_KV, HEAD_DIM), \
                f"缓存形状错误: {tuple(p[0].shape)}, 应为 (1, {t+1}, {N_KV}, {HEAD_DIM})"
    inc_out = torch.cat(outs, dim=1)
    assert torch.allclose(full_out, inc_out, atol=1e-4), \
        "增量与全量不一致: 检查 start_pos 推断和 freqs 表切片"


def test_aux_loss_collected_for_moe():
    """MoE 模型: aux_loss 等于各层之和; eval 模式下为 0"""
    model = make_model(seed=5, moe=True)
    model.train()
    ids = rand_ids(2, 3)
    _, _, aux = model(ids)
    total = sum(float(l.mlp.aux_loss) for l in model.layers)
    assert abs(float(aux) - total) < 1e-9, "aux_loss 应该等于各 MoE 层 aux_loss 之和"
    model.eval()
    _, _, aux = model(ids)
    assert float(aux) == 0.0, "eval 模式下 aux_loss 应该是 0"


def test_gradients_flow():
    """嵌入和所有层的所有参数都有非零梯度"""
    model = make_model(seed=7)
    out, _, _ = model(rand_ids(2, 3))
    out.sum().backward()
    g = model.embed_tokens.weight.grad
    assert g is not None and g.abs().sum() > 0, "embedding 没有梯度"
    for l in model.layers:
        for n, p in l.named_parameters():
            assert p.grad is not None, f"{n} 没有梯度"
            assert p.grad.abs().sum() > 0, f"{n} 梯度全零"


def test_rope_buffer_self_heals():
    """freqs_cos 被清零后, forward 应察觉异常(cos[0,0] != 1)并重算整张表"""
    model = make_model(seed=8)
    with torch.no_grad():
        model.freqs_cos.zero_()
    model(rand_ids(1, 3))
    assert torch.allclose(model.freqs_cos[0], torch.ones(HEAD_DIM), atol=1e-6), \
        "freqs_cos 被清零后, forward 应该重算 RoPE 表"


def test_hf_cache_object_ignored():
    """传入 HF 风格的 Cache 对象(带 .layers 属性)会被丢弃, 结果与无缓存一致"""
    model = make_model(seed=9)
    ids = rand_ids(1, 4)
    out1, _, _ = model(ids)

    class FakeHFCache:
        layers = []

    out2, _, _ = model(ids, past_key_values=FakeHFCache())
    assert torch.allclose(out1, out2), "带 .layers 的 Cache 对象应被丢弃, 当作无缓存处理"


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
