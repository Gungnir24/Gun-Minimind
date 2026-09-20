"""PretrainDataset 单元测试: jsonl -> 固定长度的 (input_ids, labels)

运行: python tests/test_pretrain_dataset.py
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from dataset.lm_dataset import PretrainDataset

MAXLEN = 64
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TOKENIZER_DIR = os.path.join(PROJECT_ROOT, "model")

tok = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
BOS, EOS, PAD = tok.bos_token_id, tok.eos_token_id, tok.pad_token_id  # 1, 2, 0


def write_jsonl(texts):
    """把若干文本写成临时 jsonl, 每行 {'text': ...}, 返回路径"""
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    with open(path, "w", encoding="utf-8") as f:
        for t in texts:
            f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
    return path


def make_ds(texts, max_length=MAXLEN):
    return PretrainDataset(write_jsonl(texts), tok, max_length=max_length)


def test_length():
    """__len__ == jsonl 的行数"""
    ds = make_ds(["第一行", "第二行", "第三行"])
    assert len(ds) == 3, f"长度错误: {len(ds)}"


def test_shapes_and_dtype():
    """每条样本: input_ids 和 labels 都是 [max_length] 的 long 张量"""
    ds = make_ds(["短句", "另一句"])
    ids, labels = ds[0]
    assert isinstance(ids, torch.Tensor) and isinstance(labels, torch.Tensor)
    assert ids.shape == (MAXLEN,) and labels.shape == (MAXLEN,)
    assert ids.dtype == torch.long and labels.dtype == torch.long


def test_short_text_structure():
    """短句: [bos, 文本..., eos, pad...] —— 首位 bos, 尾部全 pad"""
    ds = make_ds(["人工智能"])
    ids, _ = ds[0]
    text_len = len(tok.encode("人工智能", add_special_tokens=False))
    assert int(ids[0]) == BOS, "位置 0 应该是 bos"
    assert int(ids[1 + text_len]) == EOS, "文本结束后应该紧跟 eos"
    assert all(int(t) == PAD for t in ids[1 + text_len + 1:]), "eos 之后应该全部是 pad"


def test_labels_mask_pad():
    """labels 与 input_ids 逐位相同, 唯 pad 位置换成 -100"""
    ds = make_ds(["机器学习很有趣", "今天天气不错"])
    ids, labels = ds[1]
    real = ids != PAD
    assert torch.equal(labels[real], ids[real]), "非 pad 位置 labels 应等于 input_ids"
    assert (labels[~real] == -100).all(), "pad 位置 labels 应该是 -100"


def test_decode_roundtrip():
    """去掉首尾标记和填充, 解码能还原原文"""
    text = "预训练就是让模型学习接龙。"
    ds = make_ds([text])
    ids, _ = ds[0]
    end = int((ids == EOS).nonzero()[0])
    assert tok.decode(ids[1:end]) == text, "解码应还原原文"


def test_truncation():
    """超长文本截到 max_length-2 个, 拼上 bos/eos 恰好占满, 没有 pad"""
    long_text = "这是一段很长的文本，" * 30
    ds = make_ds([long_text], max_length=MAXLEN)
    ids, labels = ds[0]
    assert ids.shape == (MAXLEN,)
    assert int(ids[0]) == BOS
    assert int(ids[-1]) == EOS, "截断时最后一位应该正好是 eos"
    assert not (ids == PAD).any(), "占满时不应有 pad"
    assert not (labels == -100).any(), "占满时不应有被忽略的位置"


def test_empty_text():
    """空文本: 只剩 [bos, eos] + 全 pad, labels 只有前两位不是 -100"""
    ds = make_ds([""])
    ids, labels = ds[0]
    assert int(ids[0]) == BOS and int(ids[1]) == EOS
    assert (ids[2:] == PAD).all()
    assert int(labels[0]) == BOS and int(labels[1]) == EOS
    assert (labels[2:] == -100).all()


def test_dataloader_batches():
    """DataLoader 直接堆叠成 [b, max_length] —— 固定长度的意义"""
    ds = make_ds(["批量测试一", "批量测试二", "批量测试三"])
    dl = DataLoader(ds, batch_size=2, shuffle=False)
    batches = list(dl)
    assert len(batches) == 2
    assert batches[0][0].shape == (2, MAXLEN) and batches[0][1].shape == (2, MAXLEN)
    assert batches[1][0].shape == (1, MAXLEN), "最后一批应该只剩 1 条"


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
