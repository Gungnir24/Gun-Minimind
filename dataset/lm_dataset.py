import torch
from torch.utils.data import Dataset
from datasets import load_dataset


class PretrainDataset(Dataset):
    """预训练数据集 (第 11 课): 每行 {'text': ...} 的 jsonl -> 固定长度 (input_ids, labels) 对

    torch.utils.data.Dataset 的契约:
        实现 __len__ 和 __getitem__ 两个方法, DataLoader 就能自动分批、打乱、多进程加载
        (本类不碰 DataLoader 的事, 只负责"单条样本怎么变")

    __init__(data_path, tokenizer, max_length=512):
        samples = load_dataset('json', data_files=data_path, split='train')
        —— HF datasets 库: jsonl 转成 Arrow 列式表, 惰性按行读取, 几个 G 的语料不爆内存

    __getitem__(index) 五步:
        1. tokens = tokenizer(str(sample['text']),
                              add_special_tokens=False,        # 不让 tokenizer 自动加特殊标记
                              max_length=self.max_length - 2,  # -2: 给手动加的 bos/eos 留位置
                              truncation=True).input_ids      # 超长截断
           (add_special_tokens=False 是因为这份 tokenizer 不会自动加, 我们手动加 —— 顺序可控,
            而且 truncation 的长度是含自动标记算的, 手动加就必须自己留出额度)
        2. tokens = [bos_token_id] + tokens + [eos_token_id]
           —— 一个完整"文档"的首尾标记: bos 标记开头, eos 教模型"说完了"(生成时靠它停)
        3. 尾部补 pad_token_id 到 max_length
           —— 每条定长, DataLoader 的默认 collate 才能直接 stack 成 [b, max_length]
        4. input_ids = torch.tensor(..., dtype=torch.long)
        5. labels = input_ids.clone(); labels[input_ids == pad_token_id] = -100
           —— labels 与 input_ids 逐位相同, 仅 pad 位换 -100(第 8 课 forward 里
              cross_entropy 的 ignore_index=-100, pad 位置不计损)
           注意这里没有做移位! 移位是 CausalLM.forward 的职责(第 8 课), 数据集只管给原样对齐的序列

    为什么不需要 attention_mask 遮 pad:
        训练用右填充(pad 全在真实 token 之后), 因果注意力保证真实位置的 token
        看不见后面的 pad; pad 位置的预测又全被 -100 扔掉了 —— 两头都堵死, 不用遮
        (生成时的 left-padding + attention_mask 是另一回事, 那是 SFT/推理侧的活)
    """

    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset('json', data_files=data_path, split='train')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        tokens = self.tokenizer(str(sample['text']), add_special_tokens=False, max_length=self.max_length - 2, truncation=True).input_ids
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
        input_ids = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = input_ids.clone()
        labels[input_ids == self.tokenizer.pad_token_id] = -100
        return input_ids, labels
        
