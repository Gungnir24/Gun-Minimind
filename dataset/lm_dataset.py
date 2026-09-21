import json
import os
import random

import torch
from torch.utils.data import Dataset
from datasets import load_dataset, Features, Value

os.environ["TOKENIZERS_PARALLELISM"] = "false"


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


def pre_processing_chat(conversations, add_system_ratio=0.2):
    # tool use 数据完整保留不做处理
    if any(conv.get('tools') for conv in conversations): return conversations

    SYSTEM_PROMPTS = [
        "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
        "你是minimind，一个小巧但有用的语言模型。",
        "你是一个专业的AI助手，请提供有价值的回答。",
        "你是minimind，请尽力帮助用户解决问题。",
        "你是一个可靠的AI，请给出准确的回答。",
        "You are a helpful AI assistant.",
        "You are minimind, a lightweight intelligent assistant.",
        "You are a friendly chatbot. Please answer the user's questions carefully.",
        "You are a knowledgeable AI. Try your best to provide accurate information.",
        "You are minimind, a small but useful language model."
    ]
    # 概率性添加system
    if conversations[0].get('role') != 'system':
        if random.random() < add_system_ratio:
            return [{'role': 'system', 'content': random.choice(SYSTEM_PROMPTS)}] + conversations
    return conversations

def post_processing_chat(prompt_content, empty_think_ratio=0.2, remove_empty_think=None):
    # 以80%概率移除空思考标签
    if '<think>\n\n</think>\n\n' in prompt_content:
        if remove_empty_think is None:
            remove_empty_think = random.random() > empty_think_ratio
        if remove_empty_think:
            prompt_content = prompt_content.replace('<think>\n\n</think>\n\n', '')
    return prompt_content


class SFTDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        features = Features({'conversations': [{'role': Value('string'), 'content': Value('string'), 'reasoning_content': Value('string'), 'tools': Value('string'), 'tool_calls': Value('string')}]})
        self.samples = load_dataset('json', data_files=jsonl_path, split='train', features=features)
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    def create_chat_prompt(self, conversations):
        messages = []
        tools = None
        for message in conversations:
            message = dict(message)
            if message.get("role") == "system" and message.get("tools"):
                tools = json.loads(message["tools"]) if isinstance(message["tools"], str) else message["tools"]
            if message.get("tool_calls") and isinstance(message["tool_calls"], str):
                message["tool_calls"] = json.loads(message["tool_calls"])
            messages.append(message)
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            tools=tools
        )

    def generate_labels(self, input_ids):
        labels = [-100] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    labels[j] = input_ids[j]
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return labels

    def __getitem__(self, index):
        sample = self.samples[index]
        conversations = pre_processing_chat(sample['conversations'])
        prompt = self.create_chat_prompt(conversations)
        prompt = post_processing_chat(prompt)
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        labels = self.generate_labels(input_ids)
        # # === 调试打印 ===
        # print(f"\n--- Sample {index} ---")
        # for i, (x, y) in enumerate(zip(input_ids[:-1], labels[1:])):
        #     print(f"{i:3d}: X={self.tokenizer.decode([x])!r:16s} ---> Y={self.tokenizer.decode([input_ids[i+1]])!r:16s} label={y}")
        # # ================
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)
