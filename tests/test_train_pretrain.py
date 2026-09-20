"""train_pretrain 单元测试: 学习率调度 + 训练循环真的在学习 + 自动存盘/续训状态 + 训完能生成

运行: python tests/test_train_pretrain.py

被测的 train_epoch 与参考一致: 不收模型/优化器参数, 读模块全局变量
(args/model/optimizer/autocast_ctx/scaler/lm_config); 不返回 loss, 日志报的是当前步的 loss。
测试侧对应三个处理:
    1. 注入: 先把六样全局塞进 trainer.train_pretrain 模块命名空间再调用
    2. 取数: 把 tp.Logger 换成抓取函数, 从每轮最后一行日志解析 loss
       (log_interval=100 > 总步数, 每轮只在 step==iters 打一行)
    3. 路径: 参考在 train_epoch 里写死 lm_checkpoint(save_dir='../checkpoints')
       (相对运行目录), 测试把 cwd 切到临时目录的子目录 run/, 让 ../checkpoints
       落在临时目录里 —— 不然它会写进项目外的 D:\\Code\\checkpoints
"""

import os
import re
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from contextlib import nullcontext
from types import SimpleNamespace

import torch
from torch import optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from dataset.lm_dataset import PretrainDataset
from trainer.trainer_utils import get_lr
import trainer.train_pretrain as tp

TOKENIZER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model")
tok = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
SENTENCE = "人工智能改变世界。"
MAXLEN = 32
N_SAMPLES = 300
BATCH = 16
ITERS = (N_SAMPLES + BATCH - 1) // BATCH  # 19 个 step
CKPT_EPOCHS = 6  # checkpoint 测试多训几轮: 自动存盘固定落在轮末, 单步 loss 噪声大,
                 # 训到 lr 衰减到底、句子记牢后, 贪心 argmax 链才对浮点/量化误差不敏感


def make_tiny_lm(seed=0):
    """64 维两层的小模型, 词表对齐真实 tokenizer 的 6400"""
    torch.manual_seed(seed)
    cfg = MiniMindConfig(hidden_size=64, num_hidden_layers=2, vocab_size=6400,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                         dropout=0.0, flash_attn=False, max_position_embeddings=MAXLEN)
    return MiniMindForCausalLM(cfg), cfg


def write_jsonl(texts):
    import json
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    with open(path, "w", encoding="utf-8") as f:
        for t in texts:
            f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
    return path


def train_tiny(epochs, seed=0):
    """在 300 份同一句话上训练小模型。

    返回 (model, cfg, 每轮最后一步的loss, 权重文件内容, 续训状态内容)。
    权重文件由 step==iters 触发, 存到 args.save_dir; 续训状态由参考写死的
    save_dir='../checkpoints' 存到 cwd 的上一级, 所以 cwd 切到 tmpd/run。
    """
    lm, cfg = make_tiny_lm(seed=seed)
    orig_logger, orig_cwd = tp.Logger, os.getcwd()
    captured = []
    tp.Logger = lambda content: captured.append(content)  # 静音 + 从日志行取 loss
    with tempfile.TemporaryDirectory() as tmpd:
        try:
            os.makedirs(os.path.join(tmpd, 'run'))
            os.chdir(os.path.join(tmpd, 'run'))
            tp.args = SimpleNamespace(device='cpu', learning_rate=3e-3, epochs=epochs,
                                      accumulation_steps=2, grad_clip=1.0, log_interval=100,
                                      save_interval=1000, save_dir=tmpd, save_weight='test')
            tp.lm_config = cfg
            tp.model = lm
            tp.optimizer = optim.AdamW(lm.parameters(), lr=tp.args.learning_rate)
            tp.autocast_ctx = nullcontext()  # CPU 上不用混合精度
            tp.scaler = torch.cuda.amp.GradScaler(enabled=False)  # fp32 训练, scaler 直通
            ds = PretrainDataset(write_jsonl([SENTENCE] * N_SAMPLES), tok, max_length=MAXLEN)
            loader = DataLoader(ds, batch_size=BATCH, shuffle=False)
            losses = []
            for e in range(epochs):
                captured.clear()
                tp.train_epoch(e, loader, len(loader))  # 参考不返回 loss, 从日志解析
                losses.append(float(re.search(r'\bloss: ([\d.]+)', captured[-1]).group(1)))
            weights = torch.load(f'{tmpd}/test_{cfg.hidden_size}.pth', map_location='cpu')
            resume = torch.load(f'{tmpd}/checkpoints/test_{cfg.hidden_size}_resume.pth', map_location='cpu')
        finally:
            tp.Logger = orig_logger
            os.chdir(orig_cwd)
    return lm, cfg, losses, weights, resume


def test_get_lr():
    """余弦退火: 起点=全值, 终点=10%, 单调递减"""
    lr = 1e-3
    assert abs(get_lr(0, 100, lr) - lr) < 1e-12, "it=0 时应该返回原始学习率"
    assert abs(get_lr(100, 100, lr) - 0.1 * lr) < 1e-12, "it=max_it 时应该衰减到 10%"
    assert abs(get_lr(50, 100, lr) - 0.55 * lr) < 1e-12, "半程应为 55% (0.1 + 0.45)"
    vals = [get_lr(i, 100, lr) for i in range(101)]
    assert all(a >= b for a, b in zip(vals, vals[1:])), "学习率应该单调不增"


def test_loss_decreases():
    """训练真的在学: 每轮最后一步的 loss 持续下降, 且明显低于 ln(6400)≈8.76 的起点"""
    _, _, losses, _, _ = train_tiny(epochs=3)
    assert losses[0] < 8.5, f"第一轮末 loss 应已低于起点, 实际 {losses[0]:.4f}"
    assert losses[1] < losses[0], f"第二轮应低于第一轮: {losses[0]:.4f} -> {losses[1]:.4f}"
    assert losses[2] < losses[1], f"第三轮应继续下降: {losses[1]:.4f} -> {losses[2]:.4f}"


def test_checkpoint_and_generation():
    """自动存盘 -> 重载: logits 一致(半精度容差); 续训状态记录到最后位置; 能接龙出训练句"""
    lm, cfg, losses, weights, resume = train_tiny(epochs=CKPT_EPOCHS)
    assert losses[-1] < 5.0, "训练充分才有生成意义"

    # step==iters 触发的自动存盘: 权重文件 + lm_checkpoint 写到 ../checkpoints 的续训状态
    assert resume['epoch'] == CKPT_EPOCHS - 1 and resume['step'] == ITERS, \
        f"续训状态应记录最后一轮({CKPT_EPOCHS - 1})最后一步({ITERS}), 实际 {resume['epoch']}/{resume['step']}"
    assert 'optimizer' in resume, "续训状态应带 optimizer 状态(否则续训时动量丢失)"

    fresh, _ = make_tiny_lm(seed=99)  # 不同随机初始化, 排除巧合
    fresh.load_state_dict(weights, strict=False)
    fresh.eval()
    lm.eval()

    ids = torch.randint(0, 6400, (2, 10))
    with torch.no_grad():
        d = (lm(ids).logits - fresh(ids).logits).abs().max().item()
    assert d < 0.05, f"重载后 logits 应一致(半精度容差), 最大差 {d}"

    prompt = torch.tensor([tok.encode("人工智能", add_special_tokens=False)])
    with torch.no_grad():
        out = fresh.generate(prompt, do_sample=False, max_new_tokens=16,
                             temperature=1.0, top_k=0, top_p=1.0, eos_token_id=None)
    text = tok.decode(out[0][prompt.shape[1]:], skip_special_tokens=True)
    assert "改变世界" in text, f"贪心接龙应续出训练内容, 实际生成: {text!r}"


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
