
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List
from torch import Tensor
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from configs.config_Anima_350M import ModelConfig


@dataclass
class AnimaConfig(PretrainedConfig):
    model_type = "Anima_350m"

    def __init__(
            self,
            n_layers: int = 24,
            emb_dim: int = 1024,
            n_heads: int = 16,
            ff_dim: int = 4096,
            rotary_dim: int = 32,
            seq_len: int = 2048,
            vocab_size: int = 32000,
            attn_dropout: float = 0.0, 
            ff_dropout: float = 0.0,
            dropout: float = 0.0,
            use_rope: bool = True,
            rope_base: float = 10000.0,
            tie_word_embeddings: bool = True,
            activation_function: str = "swiglu",
            norm_type: str = "rmsnorm",
            **kwargs,
            ):
        super().__init__(**kwargs)
        self.n_layers = n_layers
        self.emb_dim = emb_dim
        self.n_heads = n_heads
        self.ff_dim = ff_dim
        self.rotary_dim = rotary_dim
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.tie_word_embeddings = tie_word_embeddings
        self.activation_function = activation_function
        self.norm_type = norm_type
        self.attn_dropout = attn_dropout
        self.ff_dropout = ff_dropout
        self.dropout = dropout
        self.use_rope = use_rope
        self.rope_base = rope_base

""" ----------- RMSNorm ----------- """
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weights = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # x: [*, dim]
        rms_x = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return x / rms_x * self.weights


""" ----------- SwiGLU ------------ """
class SwiGLU(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return F.silu(x1) * x2

""" ----------- Rotery Positional Embedding ----------- """
class RotaryEmbedding(nn.Module):
    def __init__(self, 
                 dim: int, 
                 base: float = 10000.0):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, seq_len: int):
        # freq = [seq_len, dim/2]
        t = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.einsum("i , j -> i j", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().unsqueeze(0)
        sin = emb.sin().unsqueeze(0)
        return cos, sin

def apply_rotary_pos_emb(q, k, cos, sin, seq_offset: int = 0):
    # q,k: (batch, seq, heads, head_dim)  --> In attention pipeline...
    b, seq, h, hd = q.shape
    cos_slice = cos[:, seq_offset: seq_offset + seq, :].to(q.device).unsqueeze(2)  # (1, seq, 1, hd)
    sin_slice = sin[:, seq_offset: seq_offset + seq, :].to(q.device).unsqueeze(2)
    def rotate_half(x):
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        x_rotate = torch.stack((-x2, x1), dim=-1).flatten(-2)
        return x_rotate
    q_rot = (q * cos_slice) + (rotate_half(q) * sin_slice)
    k_rot = (k * cos_slice) + (rotate_half(k) * sin_slice)
    return q_rot, k_rot


""" ---------- MultiHeadAttention with KV-Cache ----------- """
class MultiHeadAttention(nn.Module):
    """
    Returns: (out, (present_k, present_v)) if return_kv True, otherwise just out.
    Accepts optional past_key/past_value in shape (batch, seq_past, heads, head_dim)
    and concatenates along seq dimension to form present.
    """
    def __init__(self, 
                 emb_dim: int, 
                 n_heads: int, 
                 attn_dropout: float = 0.0, 
                 rope_base: float = 10000.0,
                 use_rope: bool = True):
        
        super().__init__()
        assert emb_dim % n_heads == 0
        self.emb_dim = emb_dim
        self.n_heads = n_heads
        self.head_dim = emb_dim // n_heads
        # scalling factor
        self.scale = 1.0 / math.sqrt(self.head_dim)
        # initialize q, k, v
        self.qkv = nn.Linear(emb_dim, 3 * emb_dim, bias=False)

        self.out_proj = nn.Linear(emb_dim, emb_dim, bias=False)
        self.dropout = nn.Dropout(attn_dropout)

        self.use_rope = use_rope
        if use_rope:
            try:
                self.rope = RotaryEmbedding(self.head_dim, base = rope_base)
            except Exception:
                self.rope = None
        else:
            self.rope = None

    def forward(self, 
                x: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None,
                past_key: Optional[torch.Tensor] = None,
                past_value: Optional[torch.Tensor] = None,
                seq_offset: int = 0,
                return_kv: bool = False):
        """
        x: (batch, seq, d_model)   -- newly provided tokens
        past_key/past_value: (batch, seq_past, n_heads, head_dim) or None
        seq_offset: used for rotary (typically seq_past)
        """
        b, seq_len, emb = x.shape
        assert emb == self.emb_dim, f"embedding dim mismatch: got {emb}, expected {self.emb_dim}"

        qkv = self.qkv(x)
        q, k, v = qkv.split(self.emb_dim, dim=-1)
        # reshape to (batch, seq, heads, head_dim)
        q = q.view(b, seq_len, self.n_heads, self.head_dim)
        k = k.view(b, seq_len, self.n_heads, self.head_dim)
        v = v.view(b, seq_len, self.n_heads, self.head_dim)

        if self.use_rope and self.rope is not None:
            # Expectation: self.rope(L) -> (cos, sin) with shapes broadcastable to (b, seq, n_heads, head_dim)
            cos, sin = self.rope(seq_len + seq_offset)
            # Find the q_rotate and k_rotate
            try:
                q, k = apply_rotary_pos_emb(q, k, cos, sin, seq_offset=seq_offset)
            except Exception:
                # If helper is unavailable, skip rotary instead of crashing so user can still run code.
                pass

        # If past exists, it should be (b, seq_past, n_heads, head_dim)
        if (past_key is None) != (past_value is None):
            raise ValueError("past_key and past_value must be both provided or both None")


        if past_key is not None and past_value is not None:
            if past_key.dim() != 4 or past_value.dim() != 4:
                raise ValueError("past_key/past_value must have shape (b, seq_past, n_heads, head_dim)")


            # Validate shapes
            assert past_key.shape[0] == b and past_value.shape[0] == b, "batch mismatch with past"
            assert past_key.shape[2] == self.n_heads and past_value.shape[2] == self.n_heads, "n_heads mismatch with past"
            assert past_key.shape[3] == self.head_dim and past_value.shape[3] == self.head_dim, "head_dim mismatch with past"


            # Concatenate along sequence dimension to form present
            k = torch.cat([past_key, k], dim=1)
            v = torch.cat([past_value, v], dim=1)

            """ 
            After concat k, v have shape: (b, seq_total, n_heads, head_dim)
            where seq_total = seq_past + seq_len
            """
        # Now compute derived sequence lengths from the tensors themselves
        seq_total = k.shape[1]

        # Protect against empty-query corner cases
        if seq_len == 0:
            # Nothing new to attend from x; return zeros of appropriate shape
            out = torch.zeros((b, 0, self.emb_dim), device=x.device, dtype=x.dtype)
            if return_kv:
                return out, (k.detach(), v.detach())
            return out
        # print(f"Shape of q is: {q.shape}")
        # print("q.numel", q.numel())
        # prepare for batched matmuls: (b*heads, seq, head_dim)
        q_ = q.transpose(1, 2).reshape(b * self.n_heads, seq_len, self.head_dim)
        k_ = k.transpose(1, 2).reshape(b * self.n_heads, seq_total, self.head_dim)  # seq_total = seq_past + seq
        v_ = v.transpose(1, 2).reshape(b * self.n_heads, seq_total, self.head_dim)

        attn_scores = torch.bmm(q_, k_.transpose(1, 2)) * self.scale  # (b*heads, seq, seq_total)

        # Apply attention mask if provided. Accept common shapes and broadcast safely.
        if attn_mask is not None:
            # attn_mask: support (b, seq_total) or (b, seq, seq_total) or (b, 1, seq, seq_total)
            # We'll convert common cases to shape (b*heads, seq, seq_total)
            if attn_mask.dim() == 2:
                # Acceptable shapes: (b, seq_total), (b, seq, seq_total), (b, 1, seq, seq_total), (b, heads, seq, seq_total)
                # (b, seq_total) -> expand to (b, heads, seq, seq_total)
                mask2 = attn_mask.unsqueeze(1).expand(-1, self.n_heads, -1)  # not exactly seq x seq_total
                # To keep it simple when using cached decoding, user should pass (b, seq, seq_total)
                # Fall back to broadcasting: expand to (b, heads, seq, seq_total) then reshape
                mask2 = mask2.unsqueeze(2)  # (b, heads, 1, seq_total)
                mask2 = mask2.expand(-1, -1, seq_len, -1)  # (b, heads, seq, seq_total)
                mask2 = mask2.reshape(b * self.n_heads, seq_len, -1)
                attn_scores = attn_scores.masked_fill(~mask2, float("-inf"))
            elif attn_mask.dim() == 3:
                # (b, seq, seq_total)
                mask2 = attn_mask.unsqueeze(1).expand(-1, self.n_heads, -1, -1)
                mask2 = mask2.reshape(b * self.n_heads, seq_len, -1)
                attn_scores = attn_scores.masked_fill(~mask2, float("-inf"))
            elif attn_mask.dim() == 4:
                # (b, 1, seq, seq_total) or (b, heads, seq, seq_total)
                mask2 = attn_mask
                if mask2.shape[1] == 1:
                    mask2 = mask2.expand(-1, self.n_heads, -1, -1)
                mask2 = mask2.reshape(b * self.n_heads, seq_len, -1)
                attn_scores = attn_scores.masked_fill(~mask2, float("-inf"))
            else:
                raise ValueError("Unsupported attn_mask dim for cached attention")

        attn = torch.softmax(attn_scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.bmm(attn, v_)
        out = out.view(b, self.n_heads, seq_len, self.head_dim).transpose(1, 2).reshape(b, seq_len, self.emb_dim)
        out = self.out_proj(out)

        if return_kv:
            # present keys/values in shape (b, seq_total, n_heads, head_dim)
            # k/v currently in shape (b, seq_total, n_heads, head_dim)
            return out, (k, v)
        return out

# --- MLP and Transformer block updated to pass caches through ---
class FeedForwardSwiGLU(nn.Module):
    def __init__(self, 
                 emb_dim: int, 
                 ff_dim: int, 
                 dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(emb_dim, ff_dim * 2)
        self.act = SwiGLU()
        self.fc2 = nn.Linear(ff_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.dropout(self.fc2(x))
        return x

""" ----------- Transformer Block ----------- """
class TransformerBlock(nn.Module):
    """
    Accepts past_kv tuple for attention and returns new present_kv.
    past_kv: tuple (past_k, past_v) or None
    present_kv returned in same format.
    """
    def __init__(self,
                 cfg: AnimaConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.emb_dim)
        self.attn = MultiHeadAttention(cfg.emb_dim, 
                                       cfg.n_heads, 
                                       cfg.attn_dropout, 
                                       cfg.rope_base, 
                                       cfg.use_rope)
        self.norm2 = RMSNorm(cfg.emb_dim)
        self.mlp = FeedForwardSwiGLU(cfg.emb_dim, 
                                     cfg.ff_dim, 
                                     cfg.ff_dropout)

    def forward(self, 
                x, 
                attn_mask: Optional[torch.Tensor] = None,
                past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                seq_offset: int = 0):
        """ 
        Attention (pre-norm) 
        """
        x_residual = x
        x = self.norm1(x)
        if past_kv is None:
            attn_out = self.attn(x, attn_mask, None, None, seq_offset=seq_offset, return_kv=False)
            # attn_out is just tensor, but we still want to compute present k/v for cache.
            # To get present kv for caching, call attention with return_kv=True
            attn_out, present_kv = self.attn(x, attn_mask, None, None, seq_offset=seq_offset, return_kv=True)
        else:
            past_k, past_v = past_kv
            attn_out, present_kv = self.attn(x, attn_mask, past_k, past_v, seq_offset=seq_offset, return_kv=True)

        x = x_residual + attn_out

        """ 
        MLP (pre-norm) 
        """
        x_residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = x + x_residual
        return x, present_kv  # present_kv = (k_total, v_total)


""" ReasoningModel with KV cache and decode_step """
class Anima(PreTrainedModel):
    config_class = AnimaConfig

    def __init__(self, 
                 config: AnimaConfig):
        super().__init__(config)
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.emb_dim)
        self.pos_scale = math.sqrt(config.emb_dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
            ])
        self.final_norm = RMSNorm(config.emb_dim)
        self.lm_head = nn.Linear(config.emb_dim, config.vocab_size)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.token_embedding.weight

        self.post_init()

    def forward(self, 
                input_ids: torch.Tensor, 
                attn_mask: Optional[torch.Tensor] = None):
        # Full-sequence forward (no caching)
        b, seq = input_ids.shape
        x = self.token_embedding(input_ids) * self.pos_scale
        if attn_mask is None:
            causal = torch.tril(torch.ones(seq, seq, dtype=torch.bool, device=x.device))
            attn_mask = causal.unsqueeze(0).expand(b, -1, -1)
        else:
            if attn_mask.dim() == 2:
                causal = torch.tril(torch.ones(seq, seq, dtype=torch.bool, device=x.device))
                attn_mask = attn_mask.unsqueeze(1) & causal.unsqueeze(0)
        seq_offset = 0
        presents = []
        for layer, block in enumerate(self.blocks):
            x, present_kv = block(x, attn_mask=attn_mask, past_kv=None, seq_offset=seq_offset)
            presents.append(present_kv)
        x = self.final_norm(x)
        logits = self.lm_head(x)      # We can use it too at the output head: F.linear(x, self.token_embedding.weight)
        return logits

    def init_kv_cache(self, 
                      batch_size: int, 
                      device: torch.device) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Creates an empty KV cache list (one tuple per layer) you can fill incrementally.
        We'll create empty tensors with shape (batch, 0, n_heads, head_dim).
        """
        cache = []
        for block in self.blocks:
            n_heads = block.attn.n_heads
            head_dim = block.attn.head_dim
            k = torch.empty((batch_size, 0, n_heads, head_dim), dtype=torch.float32, device=device)
            v = torch.empty((batch_size, 0, n_heads, head_dim), dtype=torch.float32, device=device)
            cache.append((k, v))
        return cache

    def decode_step(self,
                    input_ids: torch.Tensor,
                    past_kv: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                    attn_mask: Optional[torch.Tensor] = None):
        """
        Single-step (or short prefix) autoregressive decoding step that returns logits
        and the updated per-layer present_kv list.
        input_ids: (batch, seq_new) - can be one token or a short new prefix.
        past_kv: list of (k,v) tuples per layer, where k/v = (batch, seq_past, heads, head_dim)
                or None to do full forward (no caching).
        attn_mask: optional mask usable for caching; for decoding, you typically pass None
                   and causal behavior is applied by providing seq_offset = seq_past
        Returns:
            logits: (batch, seq_new, vocab)
            present_kv_list: list with length n_layers of (k_total, v_total)
        """
        b, seq_new = input_ids.shape
        device = input_ids.device
        x = self.token_embedding(input_ids) * self.pos_scale

        # We'll create attn_mask suitable for (past+new) where necessary.
        # For simple causal decoding, we can construct a mask of shape (b, seq_new, seq_past + seq_new)
        presents = []
        seq_past = 0
        if past_kv is not None:
            # infer seq_past from first layer's past key
            seq_past = past_kv[0][0].shape[1] if past_kv[0][0].numel() != 0 else 0

        # If attn_mask provided and is 2D (padding for full sequence), user must provide
        # correct mask for past+new length. For simple generation we skip attn_mask (causal).
        for i, block in enumerate(self.blocks):
            past = None
            if past_kv is not None:
                past = past_kv[i]
            # seq_offset = seq_past for rotary application
            x, present_kv = block(x, attn_mask=attn_mask, past_kv=past, seq_offset=seq_past)
            presents.append(present_kv)

        x = self.final_norm(x)
        logits = F.linear(x, self.token_embedding.weight)  # (b, seq_new, vocab)
        return logits, presents
    
    @torch.no_grad()
    def generate(model,
                 input_ids,
                 max_new_tokens=100,
                 temperature=1.0,
                 top_k=None,
                 top_p=None,
                 eos_token_id=None,
                 ):
        """
        Simple autoregressive generation loop with KV-cache.
        Designed to be readable & hackable for research.
        
        Args:
            model: Your LM with forward(input_ids, cache) → (logits, new_cache)
            input_ids: [1, T] starting prompt
            max_new_tokens: number of tokens to generate
            temperature: sampling temperature
            top_k: keep only top-k logits (optional)
            top_p: nucleus sampling threshold (optional)
            eos_token_id: stop if generated
        """

        model.eval()

        # KV-cache starts empty for each layer
        cache = None  

        # We allow incremental feeding: the model decides to consume only last token if cache is present.
        for _ in range(max_new_tokens):

            # Forward step — model must use cache if provided
            logits, cache = model(input_ids, cache=cache)

            # Grab only the last token logits
            logits = logits[:, -1, :]  # [1, vocab]

            # Temperature
            if temperature != 1.0:
                logits = logits / temperature

            # Top-k filtering (optional)
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                min_val = v[:, -1].unsqueeze(1)
                logits = torch.where(logits < min_val, torch.full_like(logits, -1e10), logits)

            # Top-p (nucleus) sampling
            if top_p is not None:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cumulative = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                mask = cumulative > top_p
                mask[..., 1:] = mask[..., :-1].clone()
                mask[..., 0] = False
                sorted_logits = sorted_logits.masked_fill(mask, -1e10)
                logits = torch.zeros_like(logits).scatter(1, sorted_idx, sorted_logits)

            # Sample or greedy
            if temperature == 0.0:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            # Append token
            input_ids = torch.cat([input_ids, next_token], dim=1)

            # Stop if EOS
            if eos_token_id is not None and next_token.item() == eos_token_id:
                break

        return input_ids

# --- Example usage / small demo ---
if __name__ == "__main__":

    config = AnimaConfig()
    model = Anima(config)

    device = torch.device("cpu")
    model.to(device)

    batch = 2
    # Initialize empty cache
    kv_cache = model.init_kv_cache(batch_size=batch, device=device)  # list of (k,v)

    # Suppose we have a prompt of 5 tokens to prime the model
    prompt = torch.randint(0, config.vocab_size, (batch, 5), device=device)
    # We can process the prompt in one decode_step to populate the cache
    with torch.no_grad():
        logits, kv_cache = model.decode_step(prompt, past_kv=kv_cache)
    print("After priming, logits shape:", logits.shape)
    # Now generate token-by-token for 10 steps
    generated = []
    input_tok = torch.randint(0, config.vocab_size, (batch, 1), device=device)  # pretend last token
    for step in range(10):
        logits, kv_cache = model.decode_step(input_tok, past_kv=kv_cache)
        # pick argmax token (toy); logits shape (b, 1, vocab)
        next_tok = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)  # (b,1)
        generated.append(next_tok)
        input_tok = next_tok  # feed back
    generated = torch.cat(generated, dim=1)
    print("Generated shape:", generated.shape)  # (b, 10)

    # total_params = sum(p.numel() for p in model.parameters())
    # print(f"Total number of parameters: {total_params:,}")
