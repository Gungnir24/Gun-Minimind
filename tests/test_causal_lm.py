"""MiniMindForCausalLM 单元测试: lm_head + 权值共享 + 移位 loss

运行: python tests/test_causal_lm.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn.functional as F

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

HIDDEN, N_HEADS, N_KV, HEAD_DIM = 64, 4, 2, 16
VOCAB, MAXPOS, LAYERS = 100, 64, 2


def make_cfg(tie=True, moe=False):
    return MiniMindConfig(
        hidden_size=HIDDEN, num_hidden_layers=LAYERS, vocab_size=VOCAB,
        num_attention_heads=N_HEADS, num_key_value_heads=N_KV,
        head_dim=HEAD_DIM, dropout=0.0, flash_attn=False,
        max_position_embeddings=MAXPOS, use_moe=moe, tie_word_embeddings=tie,
    )


def make_lm(seed=0, **kw):
    torch.manual_seed(seed)
    lm = MiniMindForCausalLM(make_cfg(**kw))
    lm.eval()
    return lm


def rand_ids(b, s, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    return torch.randint(0, VOCAB, (b, s))


def test_output_shape():
    """logits [b, s, vocab]; 不给 labels 则 loss 为 None; 输出对象字段齐全"""
    lm = make_lm()
    ids = rand_ids(2, 5)
    out = lm(ids)
    assert out.logits.shape == (2, 5, VOCAB), f"logits 形状错误: {tuple(out.logits.shape)}"
    assert out.loss is None, "不给 labels 时 loss 应该是 None"
    assert out.past_key_values == [None] * LAYERS
    assert float(out.aux_loss) == 0.0


def test_lm_head_and_config():
    """lm_head: hidden -> vocab, 无 bias; config_class 挂对了"""
    lm = make_lm()
    assert lm.lm_head.weight.shape == (VOCAB, HIDDEN)
    assert lm.lm_head.bias is None
    assert lm.config_class is MiniMindConfig


def test_weight_tying():
    """tie=True: 同一个 Parameter 对象, 参数量不重复计; tie=False: 各自独立, 多 vocab*hidden"""
    lm = make_lm()
    assert lm.lm_head.weight is lm.model.embed_tokens.weight, \
        "tie_word_embeddings=True 时, lm_head 和 embed_tokens 应该是同一个 Parameter 对象"
    n_tied = sum(p.numel() for p in lm.parameters())

    lm2 = make_lm(tie=False)
    assert lm2.lm_head.weight is not lm2.model.embed_tokens.weight, \
        "tie=False 时两边应该是各自独立的 Parameter"
    n_untied = sum(p.numel() for p in lm2.parameters())
    assert n_untied - n_tied == VOCAB * HIDDEN, \
        f"不共享时应多出 vocab*hidden 参数, 差值 {n_untied - n_tied}"


def test_loss_matches_manual():
    """loss 的经典移位: 位置 t 的 logits 预测 token t+1"""
    lm = make_lm(seed=42)
    ids = rand_ids(2, 5, seed=1)
    out = lm(ids, labels=ids)
    x = out.logits[..., :-1, :].contiguous()
    y = ids[..., 1:].contiguous()
    expected = F.cross_entropy(x.view(-1, VOCAB), y.view(-1), ignore_index=-100)
    assert torch.allclose(out.loss, expected, atol=1e-6), "与移位后的手工 loss 不一致"
    wrong = F.cross_entropy(out.logits.reshape(-1, VOCAB), ids.reshape(-1))
    assert not torch.allclose(out.loss, wrong), "loss 好像没做移位(用位置 t 预测了 t 自己)"


def test_ignore_index():
    """labels 里的 -100 不计损(SFT 阶段遮 prompt 就靠它)"""
    lm = make_lm(seed=3)
    ids = rand_ids(1, 5, seed=2)
    labels = ids.clone()
    labels[0, 3:] = -100
    out = lm(ids, labels=labels)
    x = out.logits[0, :-1].contiguous()
    y = labels[0, 1:].contiguous()
    expected = F.cross_entropy(x, y, ignore_index=-100)
    assert torch.allclose(out.loss, expected, atol=1e-6), "-100 的位置应该被 loss 无视"


def test_logits_to_keep():
    """logits_to_keep=1: 只算最后一个位置的 logits, 值与全量版的最后一位一致"""
    lm = make_lm(seed=4)
    ids = rand_ids(2, 5)
    out_full = lm(ids)
    out_last = lm(ids, logits_to_keep=1)
    assert out_last.logits.shape == (2, 1, VOCAB), \
        f"logits_to_keep=1 时形状错误: {tuple(out_last.logits.shape)}"
    assert torch.allclose(out_last.logits, out_full.logits[:, -1:, :], atol=1e-6)
    out_cache = lm(ids, use_cache=True, logits_to_keep=1)
    assert out_cache.past_key_values[0][0].shape == (2, 5, N_KV, HEAD_DIM), \
        "logits_to_keep 不应该影响缓存的产生"


def test_aux_loss_passthrough_for_moe():
    """MoE 模型: aux_loss 从主干一路带出来"""
    lm = make_lm(seed=6, moe=True)
    lm.train()
    ids = rand_ids(2, 4)
    out = lm(ids)
    total = sum(float(l.mlp.aux_loss) for l in lm.model.layers)
    assert abs(float(out.aux_loss) - total) < 1e-9, "aux_loss 应该原样透传"


def test_gradients_flow():
    """loss 反传: 共享权重的梯度从两条路(查表+打分)汇聚; 全部层参数有梯度"""
    lm = make_lm(seed=5)
    lm.train()
    ids = rand_ids(2, 5)
    lm(ids, labels=ids).loss.backward()
    assert lm.lm_head.weight.grad is lm.model.embed_tokens.weight.grad, \
        "权值共享时 .grad 也应该是同一个张量对象"
    assert lm.lm_head.weight.grad.abs().sum() > 0
    for l in lm.model.layers:
        for n, p in l.named_parameters():
            assert p.grad is not None, f"{n} 没有梯度"
            assert p.grad.abs().sum() > 0, f"{n} 梯度全零"


def test_save_load_roundtrip():
    """继承 PreTrainedModel 的红利: 存盘再加载, 输出一致"""
    lm = make_lm(seed=11)
    ids = rand_ids(2, 5)
    out1 = lm(ids).logits
    with tempfile.TemporaryDirectory() as d:
        lm.save_pretrained(d)
        loaded = MiniMindForCausalLM.from_pretrained(d)
    loaded.eval()
    out2 = loaded(ids).logits
    assert torch.allclose(out1, out2, atol=1e-5), "存盘再加载后输出应该一致"


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
