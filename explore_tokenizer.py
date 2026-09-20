from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("model")
print("词表长度:", len(tok))                      # 6400

ids = tok.encode("人工智能是计算机科学的分支")
print("编码:", ids)
print("逐个片段:", tok.convert_ids_to_tokens(ids))  # 常见字是单 token, 生僻词被拆碎
print("解码:", tok.decode(ids))                    # 应该原样还原
print("压缩率:", len("人工智能是计算机科学的分支") / len(ids))  # 字符数/token数, 约 1.5-2

print("bos:", tok.bos_token_id, tok.bos_token)     # 1 <|im_start|>
print("eos:", tok.eos_token_id, tok.eos_token)     # 2 <|im_end|>
print("pad:", tok.pad_token_id, tok.pad_token)     # 0

# 对话模板(SFT 阶段才用, pretrain 不用, 先混个脸熟)
msgs = [{"role": "user", "content": "你好"}, {"role": "assistant", "content": "你好！"}]
print(tok.apply_chat_template(msgs, tokenize=False))
