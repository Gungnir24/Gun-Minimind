import math
import torch
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import MoeCausalLMOutputWithPast
class MiniMindConfig(PretrainedConfig):
    model_type = "minimind"
    def __init__(self, hidden_size=768, num_hidden_layers=8, use_moe=False, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.use_moe = use_moe
        self.dropout = kwargs.get("dropout", 0.0)
        self.vocab_size = kwargs.get("vocab_size", 6400)
        self.bos_token_id = kwargs.get("bos_token_id", 1)
        self.eos_token_id = kwargs.get("eos_token_id", 2)
        self.flash_attn = kwargs.get("flash_attn", True)
        self.num_attention_heads = kwargs.get("num_attention_heads", 8)
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 4)
        self.head_dim = kwargs.get("head_dim", self.hidden_size // self.num_attention_heads)
        self.hidden_act = kwargs.get("hidden_act", 'silu')
        self.intermediate_size = kwargs.get("intermediate_size", math.ceil(hidden_size * math.pi / 64) * 64)
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 32768)
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)
        self.rope_theta = kwargs.get("rope_theta", 1e6)
        self.tie_word_embeddings = kwargs.get("tie_word_embeddings", True)
        self.inference_rope_scaling = kwargs.get("inference_rope_scaling", False)
        self.rope_scaling = {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 16,
            "original_max_position_embeddings": 2048,
            "attention_factor": 1.0,
            "type": "yarn"
        } if self.inference_rope_scaling else None
        ### MoE specific configs (ignored if use_moe = False)
        self.num_experts = kwargs.get("num_experts", 4)
        self.num_experts_per_tok = kwargs.get("num_experts_per_tok", 1)
        self.moe_intermediate_size = kwargs.get("moe_intermediate_size", self.intermediate_size)
        self.norm_topk_prob = kwargs.get("norm_topk_prob", True)
        self.router_aux_loss_coef = kwargs.get("router_aux_loss_coef", 5e-4)


"""
MiniMind 手搓版 —— 从零实现一个 Qwen3 结构的小语言模型

组件清单（按实现顺序，每完成一个就跑 tests/ 里对应的测试）:
  [x] 1. RMSNorm
  [x] 2. RoPE           旋转位置编码
  [x] 3. Attention      GQA + QK-Norm + KV Cache
  [x] 4. FeedForward    SwiGLU
  [x] 5. MOEFeedForward  稀疏专家 + 路由器
  [x] 6. Block          残差块(Pre-Norm)
  [x] 7. MiniMindModel  层堆叠 + 最终 Norm
  [x] 8. CausalLM       lm_head + 权值共享 + loss (含修复参考项目 from_pretrained 后 tie 断裂的 bug)
  [x] 9. generate       采样解码(贪心/温度/top_p)

=> 模型部分完结: 从 RMSNorm 到能生成文本的完整 Qwen3 结构 LLM, 全部手搓 + 单元测试通过
"""

class RMSNorm(nn.Module):
    """均方根归一化（Qwen3 / Llama 同款）

    公式: y = x / sqrt(mean(x^2) + eps) * weight

    - 归一化只发生在最后一个维度（每个 token 独立归一化）
    - weight: 可学习缩放参数, 形状 (dim,), 初始化为全 1
    - eps: 防止除零的小常数

    实现要求:
    1. weight 用 nn.Parameter 定义, 初始值全 1
    2. 统计量（平方均值）必须在 float32 上计算, 无论输入是什么精度
       —— 混合精度训练时, fp16 的平方和会溢出为 inf
    3. 最终输出的 dtype 与输入保持一致
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim = True) + self.eps)

    def forward(self, x):
        return (self.weight * self.norm(x.float())).type_as(x)


def precompute_freqs_cis(dim: int, end: int, rope_base: float = 1e6, rope_scaling: dict = None):
    """预计算 RoPE 的 cos/sin 角度表

    参数:
        dim: head_dim —— 每个注意力头内部的维度(注意不是 hidden_size!)
        end: 表覆盖的位置范围 [0, end)
        rope_base: 频率基数 theta, minimind 用 1e6(为长上下文准备)

    返回: (freqs_cos, freqs_sin), 形状均为 [end, dim]

    角度定义:
        - 第 j 对维度的频率: theta_j = rope_base^(-2j/dim), j = 0, 1, ..., dim/2-1
          j=0 时 theta=1(转得最快, 波长最短), j 越大转得越慢(波长越长)
        - 位置 t 的角度: t * theta_j

    布局约定 (llama 系, 与 apply_rotary_pos_emb 的 rotate_half 配套):
        freqs_cos = cat([cos(angles), cos(angles)], dim=-1)   # 前后半重复
        freqs_sin 同理
    """
    freqs, attn_factor = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)), 1.0

    if rope_scaling is not None: # YaRN: f'(i) = f(i)((1-γ) + γ/s), where γ∈[0,1] is linear ramp
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048), rope_scaling.get("factor", 16),
            rope_scaling.get("beta_fast", 32.0), rope_scaling.get("beta_slow", 1.0), rope_scaling.get("attention_factor", 1.0)
        )
        if end / orig_max > 1.0:
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
            low, high = max(math.floor(inv_dim(beta_fast)), 0), min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
            ramp = torch.clamp((torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001), 0, 1)
            freqs = freqs * (1 - ramp + ramp / factor)

    t = torch.arange(end, device = freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim = 1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim = 1) * attn_factor
    return freqs_cos, freqs_sin


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """对 q/k 施加旋转位置编码

    参数:
        q, k: [batch, seq, n_heads, head_dim]
        cos, sin: [seq, head_dim] (或 [1, head_dim] 表示单一位置)
        unsqueeze_dim: cos/sin 在第几个维度插入长度 1, 以便广播到 n_heads

    返回: (q_embed, k_embed), 形状和 dtype 都与输入相同

    公式:
        rotate_half(x) = cat(-x[..., dim/2:], x[..., :dim/2])
        q_embed = q * cos + rotate_half(q) * sin   (k 同理)

    实现要求:
    1. 只旋转 q 和 k(v 根本不会被传进来 —— 位置只应影响"匹配", 不应影响"内容")
    2. cos/sin 先 unsqueeze(unsqueeze_dim) 再与 q/k 相乘
    3. 输出 dtype 转回 q/k 各自的原始 dtype(cos/sin 是 float32)
    """
    def rotate_half(x) : return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)
    q_embed = ((q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))).to(q.dtype)
    k_embed = ((k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))).to(k.dtype)
    return q_embed, k_embed


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """GQA: 把 KV 头复制 n_rep 份, 扩到和 Q 头一样多

    x: [batch, seq, n_kv_heads, head_dim] -> [batch, seq, n_kv_heads * n_rep, head_dim]

    排列要求: n_rep=2 时, 扩出来的第 0、1 个头都来自 KV 头 0, 第 2、3 个头都来自 KV 头 1
    (即 Q 头 2i 和 2i+1 共享 KV 头 i —— 这就是"分组查询"的含义)

    提示: x[:, :, :, None, :].expand(...).reshape(...)
    """
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1: return x
    return (x[:, :, :, None, :].expand(bs, slen, num_key_value_heads, n_rep, head_dim).reshape(bs, slen, num_key_value_heads * n_rep, head_dim))


class Attention(nn.Module):
    """分组查询注意力: GQA + QK-Norm + KV Cache + causal mask

    __init__ 按初始化(记号: 4 个 Q 头, 2 个 KV 头, head_dim=16, hidden=64):
        n_local_heads    = config.num_attention_heads      # Q 头数 (4)
        n_local_kv_heads = config.num_key_value_heads      # KV 头数 (2, 比 Q 少!)
        n_rep            = Q头数 // KV头数                   # 每个 KV 头服务几个 Q 头 (2)
        head_dim         = config.head_dim
        # 四个投影, 全部 bias=False:
        q_proj: hidden -> n_heads * head_dim               # 64 -> 64
        k_proj: hidden -> n_kv_heads * head_dim            # 64 -> 32, 输出比 q 小一半!
        v_proj: hidden -> n_kv_heads * head_dim            # 64 -> 32
        o_proj: n_heads * head_dim -> hidden                # 64 -> 64
        # QK-Norm, 直接复用你写的 RMSNorm(head_dim, eps=config.rms_norm_eps)
        q_norm, k_norm
        # dropout
        attn_dropout, resid_dropout = nn.Dropout(config.dropout)
        # flash 开关
        self.flash = hasattr(F, 'scaled_dot_product_attention') and config.flash_attn
    """
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.num_key_value_heads = config.num_attention_heads if config.num_key_value_heads is None else config.num_key_value_heads
        self.n_local_heads = config.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = config.head_dim
        self.is_causal = True
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias = False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias = False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias = False)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias = False)
        self.q_norm = RMSNorm(self.head_dim, eps = config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps = config.rms_norm_eps)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and config.flash_attn

    """
    forward(x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None)
        x: [batch, seq, hidden]
        position_embeddings: (cos, sin), 各 [seq, head_dim] —— 上层已按 start_pos 切好
        past_key_value: (k_past, v_past), 各 [batch, past_len, n_kv_heads, head_dim]
        返回: (output, past_kv), output: [batch, seq, hidden]

    完整流程(b=2, s=5, hidden=64, 4Q头, 2KV头, head_dim=16):
    1. 投影分头:
         xq = q_proj(x).view(b, s, 4, 16)   # [2,5,4,16]
         xk = k_proj(x).view(b, s, 2, 16)   # [2,5,2,16]
         xv = v_proj(x).view(b, s, 2, 16]
    2. QK-Norm: xq, xk = self.q_norm(xq), self.k_norm(xk)
    3. RoPE: xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)   # v 不转!
    4. KV Cache: past_key_value 非空时, xk = cat([k_past, xk], dim=1), xv 同理
    5. past_kv = (xk, xv) if use_cache else None   # 返回 cat 之后的完整 k/v
    6. 换布局到 [b, h, s, d]: 三者都 transpose(1, 2); k/v 先 repeat_kv(xk, self.n_rep)
    7. 手动注意力路径(先写这个):
         scores = xq @ xk.transpose(-2, -1) / sqrt(head_dim)          # [2,4,5,5]
         causal: scores[:, :, :, -seq_len:] += 全 -inf 的 (s,s) 矩阵.triu(1)
         padding(若有 attention_mask):
             scores += (1 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9
         probs = softmax(scores.float(), dim=-1).type_as(xq)
         output = probs @ xv                                          # [2,4,5,16]
    8. flash 路径(最后再加): F.scaled_dot_product_attention(xq, xk, xv, is_causal=True)
    9. 收尾: output = output.transpose(1, 2).reshape(b, s, -1)
       return self.resid_dropout(self.o_proj(output)), past_kv
    """
    def forward(self, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        bsz, seq_len, _ = x.shape
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xq, xk = self.q_norm(xq), self.k_norm(xk)
        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim = 1)
            xv = torch.cat([past_key_value[1], xv], dim = 1)
        past_kv = (xk, xv) if use_cache else None
        xq, xk, xv = (xq.transpose(1, 2), repeat_kv(xk, self.n_rep).transpose(1, 2), repeat_kv(xv, self.n_rep).transpose(1, 2))
        if self.flash and (seq_len > 1) and (not self.is_causal or past_key_value is None) and (attention_mask is None or torch.all(attention_mask == 1)):
            output = F.scaled_dot_product_attention(xq, xk, xv, dropout_p=self.dropout if self.training else 0.0, is_causal=self.is_causal)
        else:
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if self.is_causal: scores[:, :, :, -seq_len:] += torch.full((seq_len, seq_len), float("-inf"), device=scores.device).triu(1)
            if attention_mask is not None: scores += (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9
            output = self.attn_dropout(F.softmax(scores.float(), dim=-1).type_as(xq)) @ xv
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv


class FeedForward(nn.Module):
    """SwiGLU 前馈网络 (Llama / Qwen3 同款)

    公式: FFN(x) = down_proj( silu(gate_proj(x)) * up_proj(x) )
           silu(t) = t * sigmoid(t)   # 平滑版 ReLU

    __init__(记号: hidden=64, intermediate=256):
        gate_proj: hidden -> intermediate    # 64 -> 256, 门控支路(决定"放行多少")
        up_proj:   hidden -> intermediate    # 64 -> 256, 内容支路(提供"被放行的东西")
        down_proj: intermediate -> hidden    # 256 -> 64, 收回 hidden 维度
        dropout = nn.Dropout(config.dropout)
        # 三个投影全部 bias=False

    forward(x): x [b, s, 64] -> [b, s, 64], 逐位置独立计算
        1. 两条支路各自投影: gate_proj(x), up_proj(x)
        2. gate 支路过激活: F.silu(...)
        3. 两支路逐元素相乘: silu(gate) * up        # [b, s, 256]
        4. down_proj 投回 hidden, 过 dropout        # [b, s, 64]

    实现要求:
    1. 激活用 F.silu(库函数), 不要手写 sigmoid 公式
    2. forward 可以一行写完
    """
    def __init__(self, config: MiniMindConfig, intermediate_size: int = None):
        super().__init__()
        intermediate_size = intermediate_size or config.intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias = False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias = False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias = False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class MOEFeedForward(nn.Module):
    """稀疏混合专家: N 个 FeedForward 专家 + 一个按 token 指路的路由器

    思想: 参数量 x N, 但每个 token 只经过 k 个专家 -> 计算量几乎不变
          (MiniMind: 4 选 1; DeepSeek-V3: 256 选 8)

    __init__(记号: hidden=64, 4 个专家, 每个专家 intermediate=256):
        config:  存一份 self.config, forward 里要读 num_experts_per_tok 等
        gate:    nn.Linear(hidden, num_experts, bias=False)   # 64 -> 4, 路由器
        experts: nn.ModuleList([FeedForward(config, intermediate_size=...) x 4])
        (参考里的 self.act_fn 是死代码: forward 从未用到, 专家自带激活, 不要抄)

    forward(x):  x [b, s, 64] -> [b, s, 64]; 辅助损失存到 self.aux_loss
        1. 摊平: x_flat = x.view(-1, hidden)             # 路由按 token 独立决策, [b*s, 64]
        2. 打分: scores = F.softmax(gate(x_flat), -1)    # [b*s, 4], 每行一个分布
        3. 选 k 个: topk_weight, topk_idx = torch.topk(scores, k, dim=-1, sorted=False)
        4. 归一(norm_topk_prob 时): topk_weight /= (topk_weight.sum(-1, keepdim=True) + 1e-20)
           —— k=1 时除完恰好 1.0: 硬路由, 选中专家全量输出
        5. 派发循环(每个专家 i 各来一遍):
             mask      = (topk_idx == i)                  # [n, k] 布尔: 哪些 token 选了我
             没单就跳过; 有单才干:
             token_idx = mask.any(-1).nonzero().flatten() # 选了我的 token 行号
             weight    = topk_weight[mask].view(-1, 1)    # 它们的路由权重
             y.index_add_(0, token_idx, expert(x_flat[token_idx]) * weight)
             —— 用 index_add_ 而非 copy: k>1 时一个 token 从多个专家收货, 要累加
        6. 空转保底(仅 training): 没接到 token 的专家,
             y[0, 0] += 0 * sum(p.sum() for p in expert.parameters())
           —— 数值贡献为 0, 但把参数拉进计算图, 保证梯度"存在", DDP 多卡训练才不卡死
        7. aux_loss(仅 training 且 router_aux_loss_coef > 0, 否则存 0):
             load = F.one_hot(topk_idx, num_experts).float().mean(0)   # 各专家流量占比
             self.aux_loss = (load * scores.mean(0)).sum() * num_experts * router_aux_loss_coef
        8. return y.view(batch_size, seq_len, hidden_dim)
    """
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias = False)
        self.experts = nn.ModuleList([FeedForward(config, intermediate_size=config.moe_intermediate_size) for _ in range(config.num_experts)])


    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.shape
        x_flat = x.view(-1, hidden_dim)
        scores = F.softmax(self.gate(x_flat), dim=-1)
        topk_weight, topk_idx = torch.topk(scores, k=self.config.num_experts_per_tok, dim=-1,sorted=False)
        if self.config.norm_topk_prob : topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True))
        y = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            mask = (topk_idx == i)
            if mask.any():
                token_idx = mask.any(dim=-1).nonzero().flatten()
                weight = topk_weight[mask].view(-1, 1)
                y.index_add_(0, token_idx, (expert(x_flat[token_idx]) * weight).to(y.dtype))
            elif self.training:
                y[0, 0] += 0 * sum(p.sum() for p in expert.parameters())
        if self.training and self.config.router_aux_loss_coef > 0:
            load = F.one_hot(topk_idx, self.config.num_experts).float().mean(0)
            self.aux_loss = (load * scores.mean(0)).sum() * self.config.num_experts * self.config.router_aux_loss_coef
        else:
            self.aux_loss = scores.new_zeros(1).squeeze()
        return y.view(batch_size, seq_len, hidden_dim)


class MiniMindBlock(nn.Module):
    """Pre-Norm 残差块: 把你写的全部零件组装成一个"阅读层"

    __init__(layer_id, config):
        layer_id: 层号, MiniMindModel 构造时会传进来(块内部用不到, 存成 self.layer_id 即可)
        self_attn                = Attention(config)
        input_layernorm          = RMSNorm(hidden_size, eps=rms_norm_eps)  # 进注意力前
        post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)  # 进 FFN 前
        mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    forward(hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        hidden_states: [b, s, hidden] —— 残差流
        position_embeddings: (cos, sin), 上层已按 start_pos 切好
        返回: (hidden_states, present_key_value), present 是注意力的 KV 缓存

        1. residual = hidden_states
        2. h, present = self_attn(input_layernorm(hidden_states), position_embeddings,
                                  past_key_value, use_cache, attention_mask)
        3. hidden_states = residual + h
        4. hidden_states = hidden_states + mlp(post_attention_layernorm(hidden_states))
        5. return hidden_states, present

    命名陷阱: post_attention_layernorm 的意思是"注意力子层之后的那个 norm",
    但它用在 FFN 之前(HuggingFace/Llama 命名习惯) —— 不是"post-norm 结构"!
    """
    def __init__(self, layer_id: int, config: MiniMindConfig):
        super().__init__()
        self.self_attn = Attention(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)


    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(self.input_layernorm(hidden_states), position_embeddings, past_key_value, use_cache, attention_mask)
        hidden_states += residual
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present_key_value


class MiniMindModel(nn.Module):
    """模型主干: 词嵌入 -> N 个 Block -> 最终 RMSNorm (lm_head 留给下一课的 CausalLM)

    __init__(config):
        embed_tokens = nn.Embedding(vocab_size, hidden_size)   # 词表里每个 id 一行
        dropout      = nn.Dropout(config.dropout)              # 嵌入后的 dropout
        layers       = nn.ModuleList([MiniMindBlock(l, config) for l in range(num_hidden_layers)])
        norm         = RMSNorm(hidden_size, eps=rms_norm_eps)  # 第 6 课埋的伏笔: 到站整理
        RoPE 表:     freqs_cos, freqs_sin = precompute_freqs_cis(head_dim, max_position_embeddings,
                                                                rope_theta, rope_scaling)
                     register_buffer("freqs_cos"/"freqs_sin", ..., persistent=False)

    forward(input_ids, attention_mask=None, past_key_values=None, use_cache=False, **kwargs):
        input_ids: [b, s] 的 token id (long 整数)
        past_key_values: 每层各自的 (k, v) 缓存组成的 list
        返回: (hidden_states [b, s, hidden], presents, aux_loss)

        1. HF 装甲: past_key_values 若是带 .layers 属性的 Cache 对象(如 DynamicCache),
           丢弃置 None —— 我们只认自己的 list 格式
        2. 没有缓存就补成 [None] * 层数
        3. start_pos = 第一层缓存的长度(past_key_values[0][0].shape[1]), 没缓存为 0
        4. hidden_states = dropout(embed_tokens(input_ids))
        5. 自愈: 若 freqs_cos[0, 0] == 0, 整表重算一遍
           —— 正常时位置 0 的 cos 恒为 1; 等于 0 说明 buffer 被 meta-device 初始化
              吞掉了(transformers>=5.x 的 from_pretrained 玩法), 用前填回
        6. position_embeddings = (freqs_cos[start_pos : start_pos + s], freqs_sin 同切片)
           —— 同一张表, 增量解码时往后切一段
        7. 逐层堆叠(zip(layers, past_key_values)):
             hidden_states, present = layer(hidden_states, position_embeddings,
                                            past_key_value=..., use_cache=..., attention_mask=...)
             presents.append(present)              # 每层各自的缓存
        8. hidden_states = self.norm(hidden_states)   # 残差流到站, 最终 norm
        9. aux_loss = 所有 MoE 层的 mlp.aux_loss 求和(dense 模型为 0)
           写法: sum(列表, 起始值), 起始值用 hidden_states.new_zeros(1).squeeze()
    """
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([MiniMindBlock(l, config) for l in range(self.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        freqs_cos, freqs_sin = precompute_freqs_cis(dim=config.head_dim, end=config.max_position_embeddings, rope_base=config.rope_theta, rope_scaling=config.rope_scaling)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, **kwargs):
        batch_size, seq_length = input_ids.shape
        if hasattr(past_key_values, 'layers') : past_key_values = None
        past_key_values = past_key_values or [None] * len(self.layers)
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        hidden_states = self.dropout(self.embed_tokens(input_ids))
        if self.freqs_cos[0, 0] == 0:
            freqs_cos, freqs_sin = precompute_freqs_cis(dim=self.config.head_dim, end=self.config.max_position_embeddings, rope_base=self.config.rope_theta, rope_scaling=self.config.rope_scaling)
            self.freqs_cos, self.freqs_sin = freqs_cos.to(hidden_states.device), freqs_sin.to(hidden_states.device)
        position_embeddings = (self.freqs_cos[start_pos:start_pos + seq_length], self.freqs_sin[start_pos:start_pos + seq_length])
        presents = []
        for layer, past_key_value in zip(self.layers, past_key_values):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)
        hidden_states = self.norm(hidden_states)
        aux_loss = sum([l.mlp.aux_loss for l in self.layers if isinstance(l.mlp, MOEFeedForward)], hidden_states.new_zeros(1).squeeze())
        return hidden_states, presents, aux_loss


class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    """在 MiniMindModel 顶上盖一层 lm_head, 得到"会接龙的完整模型" (第 8 课)

    __init__(config):
        1. self.config = config (要在 super().__init__() 之前自己挂上)
        2. super().__init__(self.config) —— PreTrainedModel 会做权重初始化前的准备
        3. self.model  = MiniMindModel(config)             # 第 7 课的主干
        4. self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        5. 权值共享(tie_word_embeddings=True 时):
             self.model.embed_tokens.weight = self.lm_head
             —— 一个 Parameter 对象, 两处名字; 训练时梯度从两条路(查表+打分)汇聚到同一份
        6. self.post_init() —— PreTrainedModel 的收尾: 初始化权重 + 调用 tie_weights()

    HF 四件套 (get/set_input/output_embeddings):
        告诉 transformers 输入嵌入是 model.embed_tokens、输出嵌入是 lm_head。
        from_pretrained 重建参数后会调用 tie_weights() 靠这四个方法重新绑定 ——
        参考实现漏了它们, 导致存盘再加载后 lm_head 变成随机初始化的孤儿
        (这是我们从 test_save_load_roundtrip 里抓到的参考项目真 bug, 你要修好它)

    forward(input_ids, attention_mask=None, past_key_values=None, use_cache=False,
            logits_to_keep=0, labels=None, **kwargs):
        1. hidden_states, past_key_values, aux_loss = self.model(...)
        2. logits = lm_head(hidden_states[:, slice(-logits_to_keep, None), :])
           —— logits_to_keep=0 时 slice(0, None) 取全部; =1 时只算最后一个位置,
              生成时每步只需要下一个 token 的分布, 省掉 (s-1)*vocab 的无用算力
        3. labels 给了才算 loss, 经典移位:
             x = logits[..., :-1, :]   # 位置 t 的输出
             y = labels[..., 1:]      # 位置 t+1 的目标
             loss = F.cross_entropy(x.view(-1, vocab), y.view(-1), ignore_index=-100)
           —— 位置 t 预测 t+1; ignore_index=-100 让 SFT 时能遮住 prompt 部分
        4. return MoeCausalLMOutputWithPast(loss=..., aux_loss=..., logits=...,
                                            past_key_values=..., hidden_states=...)
           —— 一个输出对象, MoE 的 aux_loss 一路带出来给训练脚本加进总 loss
    """
    config_class = MiniMindConfig
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: MiniMindConfig = None):
        self.config = config or MiniMindConfig()
        super().__init__(self.config)
        self.model = MiniMindModel(self.config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        if self.config.tie_word_embeddings: self.model.embed_tokens.weight = self.lm_head.weight
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(self, input_ids, attention_mask=None, past_key_values=None,
                use_cache=False, logits_to_keep=0, labels=None, **kwargs):
        hidden_states, past_key_values, aux_loss = self.model(input_ids, attention_mask, past_key_values, use_cache, **kwargs)
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        loss = None
        if labels is not None:
            x, y = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()
            loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)
        return MoeCausalLMOutputWithPast(loss=loss, aux_loss=aux_loss, logits=logits, past_key_values=past_key_values, hidden_states=hidden_states)

    @torch.inference_mode()
    def generate(self, inputs=None, attention_mask=None, max_new_tokens=8192,
                 temperature=0.85, top_p=0.85, top_k=50, eos_token_id=2,
                 streamer=None, use_cache=True, num_return_sequences=1,
                 do_sample=True, repetition_penalty=1.0, **kwargs):
        """自回归生成 (第 9 课): 循环"forward -> 取最后一位分布 -> 选出 1 个 token -> 拼回输入"

        @torch.inference_mode(): 生成不需要梯度, 关掉 autograd 省内存提速度

        循环前的准备:
            1. input_ids = inputs.repeat(num_return_sequences, 1)
               —— 同一个 prompt 复制 N 份, 每份独立采样, 得到 N 个候选回答
            2. finished = 全 False 的 bool 向量, 每个序列一个 —— 谁生成了 eos 谁提前完工
            3. streamer 不为空就先 put 一次完整 prompt (流式显示用)

        循环体内每步 (max_new_tokens 次):
            1. past_len = past_key_values[0][0].shape[1] (没缓存为 0)
               outputs = self.forward(input_ids[:, past_len:], ...)
               —— 只把"新增的 token"喂进模型, 前面全部走 KV 缓存;
                  use_cache=False 时 past 每步清空, past_len 恒为 0, 等价全量重算
            2. attention_mask 右侧拼一列 1 (新 token 对所有旧 token 可见)
            3. logits = outputs.logits[:, -1, :] / temperature
               —— 只要最后一个位置的分布; 除温度: T<1 拉大分差(保守), T>1 压平分差(发散)
            4. 三个过滤器依次套在 logits 上:
               a) repetition_penalty != 1 时, 对"已生成过的 token"降分:
                  正分除以 penalty, 负分乘以 penalty (CTRL 论文的公式, 正负分开处理)
               b) top_k > 0 时: 只保留分数最高的 top_k 个, 其余设 -inf
               c) top_p < 1.0 时(核采样): 按分数降序排列, 累计概率超过 top_p 的尾部设 -inf
                  注意右移一位 —— 保证概率最高的那个 token 永远不会被滤掉
            5. 选 token:
               do_sample=True:  torch.multinomial(softmax(logits)) 按概率抽样
               do_sample=False: torch.argmax 直接取最大 (贪心, 确定性)
            6. 已完工的序列强制产出 eos_token_id (保持 batch 矩形不塌)
            7. input_ids = cat([input_ids, next_token]); 更新 past_key_values
            8. finished |= 新 token 是 eos; 全部完工则 break

        收尾:
            streamer.end()
            kwargs 里有 return_kv=True 时返回 {'generated_ids': input_ids, 'past_kv': past_key_values}
            (注意: 返回的缓存比 ids 落后一步 —— 最后那个 token 还没进过模型,
             它的用途是作为下一次调用的输入)
            否则只返回 input_ids
        """
        input_ids = kwargs.pop("input_ids", inputs).repeat(num_return_sequences, 1)
        attention_mask = attention_mask.repeat(num_return_sequences, 1) if attention_mask is not None else None
        past_key_values = kwargs.pop("past_key_values", None)
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        if streamer: streamer.put(input_ids.cpu())
        for _ in range(max_new_tokens):
            past_len = past_key_values[0][0].shape[1] if past_key_values else 0
            outputs = self.forward(input_ids[:, past_len:], attention_mask, past_key_values, use_cache=use_cache, **kwargs)
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones(attention_mask.shape[0], 1)], -1) if attention_mask is not None else None
            logits = outputs.logits[:, -1, :] / temperature
            if repetition_penalty != 1.0:
                for i in range(input_ids.shape[0]):
                    seen = torch.unique(input_ids[i]); score = logits[i, seen]; logits[i, seen] = torch.where(score > 0, score / repetition_penalty, score * repetition_penalty)
            if top_k > 0: 
                logits[logits < torch.topk(logits, top_k)[0][..., -1, None]] = -float('inf')
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                mask = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1) > top_p
                mask[..., 1:], mask[..., 0] = mask[..., :-1].clone(), 0
                logits[mask.scatter(1, sorted_indices, mask)] = -float('inf')
            next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1) if do_sample else torch.argmax(logits, dim=-1, keepdim=True)
            if eos_token_id is not None: next_token = torch.where(finished.unsqueeze(-1), next_token.new_full((next_token.shape[0], 1), eos_token_id), next_token)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            past_key_values = outputs.past_key_values if use_cache else None
            if streamer: streamer.put(next_token.cpu())
            if eos_token_id is not None:
                finished |= next_token.squeeze(-1).eq(eos_token_id)
                if finished.all(): break
        if streamer: streamer.end()
        if kwargs.get("return_kv"): return {'generated_ids': input_ids, 'past_kv': past_key_values}
        return input_ids

 

