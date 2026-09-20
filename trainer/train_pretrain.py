"""MiniMind 手搓版预训练脚本 (第 12 课): 与参考实现保持一致的结构

与参考版本的 3 处差异(其余 1:1):
    1. 路径默认值按"从项目根目录运行"调整: save_dir=./out, data_path=./dataset/..., tokenizer=./model
       (参考的 ../out 等假定从 trainer/ 目录里运行)
    2. train_epoch 返回本轮平均 loss (参考无返回值; 加了供测试和日志使用)
    3. --num_workers 默认 0 (参考是 8; Windows 上多进程加载易出问题, Linux 可改回)

运行: python trainer/train_pretrain.py --data_path <jsonl> --epochs 2
"""

import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import datasets  # noqa: F401  # Windows pyarrow/torch DLL conflict workaround (issue #771)
import argparse
import time
import warnings
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from model.model_minimind import MiniMindConfig
from dataset.lm_dataset import PretrainDataset
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler

warnings.filterwarnings('ignore')


def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    """跑完一个 epoch 并返回平均 loss。
    参考实现的风格: 本函数不收模型/优化器参数, 直接读模块全局变量
    (args / model / optimizer / autocast_ctx / scaler / lm_config, 由 __main__ 准备好)。
    参数:
        epoch: 当前轮号(0 起), 用于学习率进度和日志
        loader: DataLoader, 每个元素是 (input_ids, labels), 都要搬到 args.device
        iters: 本轮总 step 数 (= len(loader); 续训跳步时 = len(loader)+start_step)
        start_step: 续训时已跑过的 step 数, 本轮 step 编号从 start_step+1 开始
        wandb: None 或 swanlab 模块 (主进程且 --use_wandb 时)
    对 loader 的每个 step:
        1. lr = get_lr(epoch*iters + step, args.epochs*iters, args.learning_rate)
           (用全局 step 走余弦调度), 写回 optimizer.param_groups 每一组的 'lr'
        2. autocast_ctx 下前向: res = model(input_ids, labels=labels)
           loss = (res.loss + res.aux_loss) / args.accumulation_steps
           (aux_loss 是 MoE 路由均衡损失, dense 模型恒为 0; 梯度累积除 N)
        3. scaler.scale(loss).backward()  (fp32 训练时 scaler 直通, 代码不用分支)
        4. 每 accumulation_steps 步才更新一次 (累积 N 个小 batch 等价一个 N 倍大 batch):
           scaler.unscale_(optimizer) -> clip_grad_norm_(model.parameters(), args.grad_clip)
           -> scaler.step(optimizer) -> scaler.update() -> optimizer.zero_grad(set_to_none=True)
           顺序: unscale_ 必须在 clip 前 (clip 要看真实梯度大小), step 必须在 unscale_ 后
        5. 日志: 每 log_interval 步或最后一步, 打印轮内平均 loss / logits_loss / aux_loss /
           lr / 预计剩余分钟; wandb 非空时同步 wandb.log
        6. 存盘 (仅主进程, 每 save_interval 步或最后一步):
           a. 权重: {args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{_moe?}.pth,
              每个张量 v.half().cpu() 后存; 取 state_dict 前剥掉 DDP(.module)/compile(_orig_mod) 壳
           b. lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer,
              scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir=args.save_dir)
              追加保存可续训状态 (含 optimizer/scaler/进度)
           存盘前 model.eval(), 存完 model.train()
    epoch 末尾: 若最后一个 step 没凑满 accumulation_steps, 把残余梯度更新掉 (否则这步白算)
    返回: total_loss / count (本轮所有 step 的平均 loss, res.loss+res.aux_loss 口径)
    """
    start_time = time.time()
    last_step = start_step
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        last_step = step
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with autocast_ctx:
            res = model(input_ids, labels=labels)
            loss = res.loss + res.aux_loss
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()

        if step % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
            current_logits_loss = current_loss - current_aux_loss
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min')
            if wandb: wandb.log({"loss": current_loss, "logits_loss": current_logits_loss, "aux_loss": current_aux_loss, "learning_rate": current_lr, "epoch_time": eta_min})

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints')
            model.train()
            del state_dict

        del input_ids, labels, res, loss

    if last_step > start_step and last_step % args.accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Pretraining")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='pretrain', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=340, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--seed', default=42, type=int, help="随机种子（DDP下每个rank为seed+rank，每轮为seed+epoch）")
    parser.add_argument("--data_path", type=str, default="../dataset/pretrain_t2t_mini.jsonl", help="预训练数据路径")
    parser.add_argument('--from_weight', default='none', type=str, help="基于哪个权重训练，为none则从头开始")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    args = parser.parse_args()
    # ========== 1. 初始化环境和随机种子 ==========
    # local_rank = init_distributed_mode()  (没设 RANK 环境变量时返回 0, 即单机模式)
    # 进了 DDP 就把 args.device 改成 f"cuda:{local_rank}"
    # setup_seed(args.seed + rank): 各卡种子不同 (rank = dist.get_rank(), 未初始化按 0)
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(args.seed + (dist.get_rank() if dist.is_initialized() else 0))
 
    # ========== 2. 配置目录、模型参数、检查ckp ==========
    # 建 args.save_dir; MiniMindConfig(hidden_size, num_hidden_layers, use_moe)
    # from_resume==1 时 ckp_data = lm_checkpoint(lm_config, weight=args.save_weight,
    #   save_dir=args.save_dir) —— model 参数不传就是"加载模式", 没有续训文件返回 None
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None
    
    # ========== 3. 设置混合精度 ==========
    # device_type = "cuda"/"cpu" 由 args.device 判断; dtype = bf16/fp16 由 args.dtype 选
    # autocast_ctx: cpu 上 nullcontext() 占位, cuda 上 torch.autocast(device_type, dtype)
    # scaler = GradScaler(enabled=(args.dtype=="float16" 且 cuda)) —— bf16 不需要 scaler
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
 
    # ========== 4. 配wandb ==========
    # 主进程且 use_wandb 才 import swanlab as wandb 并 init;
    # ckp_data 里有 wandb_id 时带 id + resume='must' 续接上次的 run
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
 
    # ========== 5. 定义模型、数据、优化器 ==========
    # model, tokenizer = init_model(lm_config, args.from_weight, tokenizer_path='./model',
    #                              save_dir=args.save_dir, device=args.device)
    #   from_weight='none' 从头训; 否则从 {save_dir}/{from_weight}_{hidden}{_moe?}.pth 加载
    # train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    # DDP 初始化时用 DistributedSampler(train_ds) 自动按 rank 切分数据
    # optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
 
    # ========== 6. 从ckp恢复状态 ==========
    # ckp_data 非空: model/optimizer/scaler 各自 load_state_dict(ckp_data[...]),
    # start_epoch = ckp_data['epoch'], start_step = ckp_data.get('step', 0)
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
 
    # ========== 7. 编译和分布式包装 ==========
    # use_compile==1: model = torch.compile(model)
    # dist 初始化过: model = DistributedDataParallel(model, device_ids=[local_rank])
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])
 
    # ========== 8. 开始训练 ==========
    # for epoch in range(start_epoch, args.epochs):
    #   train_sampler 存在时 set_epoch(epoch) —— DDP 各卡每轮用同样的打乱顺序
    #   setup_seed(args.seed + epoch) 固定本轮数据顺序 (randperm 可复现, 续训才能跳步)
    #   indices = torch.randperm(len(train_ds)).tolist()  (单机时的打乱)
    #   skip = start_step (仅当本轮是续训的第一轮且 start_step>0), 否则 0
    #   batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
    #   loader = DataLoader(train_ds, batch_sampler=batch_sampler,
    #                       num_workers=args.num_workers, pin_memory=True)
    #   train_epoch(epoch, loader, len(loader)+skip, skip, wandb)  # iters 含被跳过的部分
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(args.seed + epoch); indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0: 
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + skip, start_step, wandb)
        else:
            train_epoch(epoch, loader, len(loader), 0, wandb)
 
    # ========== 9. 清理分布进程 ==========
    # dist 初始化过才 dist.barrier() + dist.destroy_process_group()
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()