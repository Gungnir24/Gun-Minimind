"""generate 单元测试: 自回归循环 + 采样(贪心/温度/top_k/top_p) + eos 停止 + 流式

运行: python tests/test_generate.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

HIDDEN, N_HEADS, N_KV, HEAD_DIM = 64, 4, 2, 16
VOCAB, MAXPOS, LAYERS = 100, 64, 2


def make_lm(seed=0):
    torch.manual_seed(seed)
    cfg = MiniMindConfig(
        hidden_size=HIDDEN, num_hidden_layers=LAYERS, vocab_size=VOCAB,
        num_attention_heads=N_HEADS, num_key_value_heads=N_KV,
        head_dim=HEAD_DIM, dropout=0.0, flash_attn=False,
        max_position_embeddings=MAXPOS,
    )
    lm = MiniMindForCausalLM(cfg)
    lm.eval()
    return lm


def rand_ids(b, s, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    return torch.randint(0, VOCAB, (b, s))


def greedy(lm, prompt, n):
    """手工贪心: 每步全量 forward(不带缓存), 取最后一位 argmax 拼回去"""
    ids = prompt.clone()
    with torch.no_grad():
        for _ in range(n):
            nxt = lm(ids).logits[:, -1, :].argmax(dim=-1, keepdim=True)
            ids = torch.cat([ids, nxt], dim=-1)
    return ids


def test_output_shape_and_prompt():
    """返回 [b, s+n] 的 id 张量; 开头原样保留 prompt"""
    lm = make_lm(seed=0)
    prompt = rand_ids(1, 3, seed=1)
    out = lm.generate(prompt, do_sample=False, max_new_tokens=4,
                      temperature=1.0, top_k=0, top_p=1.0, eos_token_id=None)
    assert isinstance(out, torch.Tensor), "默认应返回张量(不是 dict)"
    assert out.shape == (1, 7), f"形状错误: {tuple(out.shape)}"
    assert torch.equal(out[:, :3], prompt), "前 3 个位置应该是原封不动的 prompt"


def test_max_new_tokens_limit():
    """eos_token_id=None 时不提前停, 恰好生成 n 个"""
    lm = make_lm(seed=1)
    prompt = rand_ids(2, 3, seed=2)
    out = lm.generate(prompt, do_sample=False, max_new_tokens=5,
                      temperature=1.0, top_k=0, top_p=1.0, eos_token_id=None)
    assert out.shape == (2, 8)


def test_greedy_matches_manual():
    """贪心 = 手工循环(每步全量 forward 取 argmax)"""
    lm = make_lm(seed=42)
    prompt = rand_ids(1, 3, seed=3)
    out = lm.generate(prompt, do_sample=False, max_new_tokens=6,
                      temperature=1.0, top_k=0, top_p=1.0, eos_token_id=None,
                      use_cache=False)
    assert torch.equal(out, greedy(lm, prompt, 6))


def test_cache_equals_nocache():
    """use_cache=True 和 False 的贪心结果一致(验证只喂新 token 的切片逻辑)"""
    lm = make_lm(seed=4)
    prompt = rand_ids(1, 3, seed=5)
    kw = dict(do_sample=False, max_new_tokens=6, temperature=1.0,
              top_k=0, top_p=1.0, eos_token_id=None)
    assert torch.equal(lm.generate(prompt, use_cache=True, **kw),
                       lm.generate(prompt, use_cache=False, **kw))


def test_top_k_1_equals_greedy():
    """top_k=1: 只留最高分, 采样被逼成贪心"""
    lm = make_lm(seed=6)
    prompt = rand_ids(1, 3, seed=7)
    kw = dict(max_new_tokens=5, temperature=1.0, top_p=1.0, eos_token_id=None)
    torch.manual_seed(0)
    sampled = lm.generate(prompt, do_sample=True, top_k=1, **kw)
    greedy_out = lm.generate(prompt, do_sample=False, top_k=0, **kw)
    assert torch.equal(sampled, greedy_out)


def test_top_p_tiny_equals_greedy():
    """top_p 极小: 核采样只留概率最高者, 等价贪心"""
    lm = make_lm(seed=8)
    prompt = rand_ids(1, 3, seed=9)
    kw = dict(max_new_tokens=5, temperature=1.0, top_k=0, eos_token_id=None)
    torch.manual_seed(0)
    sampled = lm.generate(prompt, do_sample=True, top_p=1e-9, **kw)
    greedy_out = lm.generate(prompt, do_sample=False, **kw)
    assert torch.equal(sampled, greedy_out)


def test_low_temperature_approximates_greedy():
    """temperature=0.01: softmax 变得极尖, 采样几乎等于贪心"""
    lm = make_lm(seed=10)
    prompt = rand_ids(1, 3, seed=11)
    kw = dict(max_new_tokens=5, top_k=0, top_p=1.0, eos_token_id=None)
    torch.manual_seed(0)
    sampled = lm.generate(prompt, do_sample=True, temperature=0.01, **kw)
    greedy_out = lm.generate(prompt, do_sample=False, temperature=1.0, **kw)
    assert torch.equal(sampled, greedy_out)


def test_eos_stops_early():
    """把贪心第一步会生成的 token 设为 eos: 生成 1 个 token 就停"""
    lm = make_lm(seed=12)
    prompt = rand_ids(1, 3, seed=13)
    with torch.no_grad():
        first = int(lm(prompt).logits[:, -1, :].argmax(dim=-1)[0])
    out = lm.generate(prompt, do_sample=False, max_new_tokens=10,
                      temperature=1.0, top_k=0, top_p=1.0, eos_token_id=first)
    assert out.shape == (1, 4), f"应在生成 1 个 token 后停止, 形状: {tuple(out.shape)}"
    assert int(out[0, 3]) == first


def test_batch_generation():
    """batch 每行各自接龙, 各行开头是各自的 prompt"""
    lm = make_lm(seed=14)
    prompts = rand_ids(2, 3, seed=15)
    out = lm.generate(prompts, do_sample=False, max_new_tokens=4,
                      temperature=1.0, top_k=0, top_p=1.0, eos_token_id=None)
    assert out.shape == (2, 7)
    assert torch.equal(out[:, :3], prompts)


def test_num_return_sequences():
    """num_return_sequences=3: 同一 prompt 复制 3 份并行采样"""
    lm = make_lm(seed=16)
    prompt = rand_ids(1, 3, seed=17)
    torch.manual_seed(0)
    out = lm.generate(prompt, do_sample=True, max_new_tokens=4,
                      temperature=1.0, top_k=0, top_p=1.0,
                      eos_token_id=None, num_return_sequences=3)
    assert out.shape == (3, 7)


def test_repetition_penalty_smoke():
    """repetition_penalty != 1 时能跑通, 输出形状正常"""
    lm = make_lm(seed=18)
    prompt = rand_ids(1, 3, seed=19)
    out = lm.generate(prompt, do_sample=False, max_new_tokens=3,
                      temperature=1.0, top_k=0, top_p=1.0,
                      eos_token_id=None, repetition_penalty=1.5)
    assert out.shape == (1, 6)
    assert torch.equal(out[:, :3], prompt)


def test_streamer():
    """streamer: 先收整个 prompt, 每步收 1 个新 token, 结束时 end()"""
    lm = make_lm(seed=20)

    class ListStreamer:
        def __init__(self):
            self.chunks, self.ended = [], False

        def put(self, ids):
            self.chunks.append(ids)

        def end(self):
            self.ended = True

    s = ListStreamer()
    prompt = rand_ids(1, 3, seed=21)
    lm.generate(prompt, do_sample=False, max_new_tokens=3,
                temperature=1.0, top_k=0, top_p=1.0,
                eos_token_id=None, streamer=s)
    assert s.ended, "结束时应该调用 end()"
    assert len(s.chunks) == 1 + 3, "应收到 1 次 prompt + 每步 1 次"
    assert s.chunks[0].shape == (1, 3)
    assert all(c.shape == (1, 1) for c in s.chunks[1:])


def test_return_kv():
    """return_kv=True: 额外带回 KV 缓存; 缓存落后 ids 一步
    (最后生成的 token 还没进过模型, 它是下一次调用的输入, 所以缓存长度 = 总长 - 1)"""
    lm = make_lm(seed=22)
    prompt = rand_ids(1, 3, seed=23)
    out = lm.generate(prompt, do_sample=False, max_new_tokens=2,
                      temperature=1.0, top_k=0, top_p=1.0,
                      eos_token_id=None, return_kv=True)
    assert isinstance(out, dict), "return_kv=True 应返回 dict"
    assert out['generated_ids'].shape == (1, 5)
    past = out['past_kv']
    assert len(past) == LAYERS
    for k, v in past:
        assert k.shape == (1, 4, N_KV, HEAD_DIM), f"k 缓存形状错误: {tuple(k.shape)}"
        assert v.shape == (1, 4, N_KV, HEAD_DIM), f"v 缓存形状错误: {tuple(v.shape)}"


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
