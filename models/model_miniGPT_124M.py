from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import random
import os
import numpy as np

import tiktoken

@dataclass
class miniGPTConfig:
    vocab_size: int = 32000
    seq_len: int = 1024
    emb_dim: int = 768
    n_heads: int = 12
    n_layers: int = 12
    drop_rate: float = 0.1
    qkv_bias: bool = False

class TransformerBlock(nn.Module):
    def __init__(self, config: miniGPTConfig):
        super().__init__()
        self.attn = nn.MultiheadAttention(config.emb_dim, config.n_heads, batch_first=True)
        self.norm_1 = nn.LayerNorm(config.emb_dim)
        self.ff = nn.Sequential(
            nn.Linear(config.emb_dim, 4*config.emb_dim),
            nn.GELU(),
            nn.Linear(4*config.emb_dim, config.emb_dim)
        )
        self.norm_2 = nn.LayerNorm(config.emb_dim)


    def forward(self, x):
        """
        x = token_emb + pos_emb
        x = [b, t, emb_dim],  t = seq_len or context_len
        """
        attn_out, _ = self.attn(x, x, x, attn_mask=self.causal_mask(x))

        """
        nn.MultiheadAttention takes three things: x=Query (Q), x=Key (K), x=Value (V).

        Here all are x because we’re doing self-attention.

      🔹 What happens inside:
        1. For each head, we create:

                   Q=x*Wq, K=x*Wk, V=x*Wv
           Shapes: [B, T, d_model] → [B, T, d_k] for each head.

        2. Compute attention scores:

             score = Q*K.T/sqrt(d_k)

        3. Apply causal mask:

            self.causal_mask(x) creates an upper-triangular mask to hide future tokens.

            Mathematically sets scores for future positions to −∞.

        4. Softmax over scores to get weights:

              weights = softmax(masked score)

        5. Weighted sum of values:

              attn_out=weights⋅V

        6. Concatenate all heads and project back to d_model.
        """
        x = x + attn_out   # Residual connection
        x = self.norm_1(x)
        x = x + self.ff(x)
        """
        x*W1+b1 -> GeLU -> x*W2+b2
        """
        x = self.norm_2(x)
        return x   # x -> [b, t, emb_dim] or [batch, context_len, emb_dim]


    def causal_mask(self, x):
        t = x.size(1)
        mask = torch.triu(torch.ones(t, t), diagonal=1).bool().to(x.device)
        return mask

class miniGPT(nn.Module):
    def __init__(self, config: miniGPTConfig):
        super().__init__()
        self.token_emb = nn.Embedding(config.vocab_size, config.emb_dim)
        self.pos_emb = nn.Embedding(config.seq_len, config.emb_dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
            ])
        self.norm_f = nn.LayerNorm(config.emb_dim)
        self.head = nn.Linear(config.emb_dim, config.vocab_size, bias=False)    # logits=h*W.T


    def forward(self, idx, use_cache=False, past_key_values=None):  # input -> idx = [batch_size, sequence_length]
        b, t = idx.shape       # b -> batch, t -> seq_len
        pos = torch.arange(0, t, device=idx.device).unsqueeze(0)
        x = self.token_emb(idx) + self.pos_emb(pos)
        """
        self.token_emb(idx) → maps token IDs to vectors of size d_model.
            * Shape: [B, T, emb_dim]
        self.pos_emb(pos) → adds position vectors.
        Summation gives initial hidden states, which are -
             input to Transformer block:

                x=h0=E[xt]+Pt  -> token_emb + pos_emb

        """
        for block in self.blocks:
            x = block(x)
            """
            Data goes through each Transformer block sequentially.
            Each block contains:
                 1. Masked self-Attention
                      Attn. = softmax(Q*K.T/sqrt(k_dim)+M)*V
                      x=x+attn.(x)
                 2. Feed-Forward Network
                      Linear + GELU + Linear
                      x=x+MLP(x)

            """
        x = self.norm_f(x)
        logits = self.head(x)   # y_pred or logits = softmax(x*W.T)
        """
        Projects from emb_dim -> vocab_size
        o/p logits shape [b, t, vocab_size]
        Each position has a distribution over possible next tokens.
        """
        return logits   # logits = [batch, context_len, vocab_size]

config = miniGPTConfig()
model = miniGPT(config)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params:,}")
print("Parameter count mostly depends on vocab_size, n_layers, seq_len, and emb_dims")

