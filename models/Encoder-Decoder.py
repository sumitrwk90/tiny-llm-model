import torch
import torch.nn as nn
import math
import torch.optim as optim
import random

device = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------- Step by Step Implementation -------------------------- #
"""
Prepare data: parallel pairs (src, tgt). Tokenize with SentencePiece/BPE.

Build model: use nn.Transformer or transformers library (T5/BART) if you want production-grade.

Training loop:

Input: src_ids, tgt_input_ids (shifted right), labels (tgt_ids)

Forward: compute logits, compute cross-entropy loss (ignore pad token)

Optimizer: AdamW, lr schedule with warmup

Use mixed precision and gradient accumulation for memory limits

Validation: decode with beam search, compute BLEU/ROUGE.

Inference: implement beam search or sampling with length penalty, repetition penalty, etc.

Fine-tuning / Transfer: start from pretrained BART/T5 weights for much faster, higher-quality results.

"""


# ============================================================
# Positional Encoding
# ============================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # shape: (1, max_len, d_model)

    def forward(self, x):
        # x: (B, T, d_model)
        x = x + self.pe[:, :x.size(1)]
        return x


# ============================================================
# Multi-Head Attention
# ============================================================
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.fc_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        B, Q_len, _ = query.size()
        B, K_len, _ = key.size()

        # Linear projections
        Q = self.W_q(query).view(B, Q_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(B, K_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(B, K_len, self.n_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(B, Q_len, -1)

        return self.fc_out(context)


# ============================================================
# Feed Forward Layer
# ============================================================
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))


# ============================================================
# Encoder Layer
# ============================================================
class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        # Self-attention
        attn_out = self.self_attn(x, x, x, mask=src_mask)
        x = self.norm1(x + self.dropout(attn_out))

        # Feedforward
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x


# ============================================================
# Decoder Layer (with Cross-Attention)
# ============================================================
class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        # 1. Decoder self-attention
        _x = self.self_attn(x, x, x, mask=tgt_mask)
        x = self.norm1(x + self.dropout(_x))

        # 2. Cross attention (query from decoder, key/value from encoder)
        _x = self.cross_attn(x, enc_out, enc_out, mask=src_mask)
        x = self.norm2(x + self.dropout(_x))

        # 3. Feed-forward
        _x = self.ff(x)
        x = self.norm3(x + self.dropout(_x))
        return x


# ============================================================
# Full Encoder
# ============================================================
class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, dropout=0.1, max_len=512):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src, src_mask=None):
        x = self.embed(src)
        x = self.pos_enc(x)
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)


# ============================================================
# Full Decoder
# ============================================================
class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, dropout=0.1, max_len=512):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, tgt, enc_out, src_mask=None, tgt_mask=None):
        x = self.embed(tgt)
        x = self.pos_enc(x)
        for layer in self.layers:
            x = layer(x, enc_out, src_mask, tgt_mask)
        x = self.norm(x)
        return self.fc_out(x)


# ============================================================
# Full Transformer Seq2Seq
# ============================================================
class TransformerSeq2Seq(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model=512, n_layers=6, n_heads=8, d_ff=2048, dropout=0.1, max_len=512):
        super().__init__()
        self.encoder = Encoder(src_vocab, d_model, n_layers, n_heads, d_ff, dropout, max_len)
        self.decoder = Decoder(tgt_vocab, d_model, n_layers, n_heads, d_ff, dropout, max_len)

    def make_tgt_mask(self, tgt):
        # Create causal mask for decoder self-attention
        T = tgt.size(1)
        mask = torch.tril(torch.ones(T, T)).unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)
        return mask.to(tgt.device)

    def forward(self, src, tgt, src_mask=None):
        enc_out = self.encoder(src, src_mask)
        tgt_mask = self.make_tgt_mask(tgt)
        logits = self.decoder(tgt, enc_out, src_mask, tgt_mask)
        return logits


# -------------------------- Dataset --------------------------- #

class ToyDataset(torch.utils.data.Dataset):
    def __init__(self, vocab_size=20, seq_len=5, size=1000):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.data = []
        for _ in range(size):
            seq = torch.randint(2, vocab_size, (seq_len,))  # avoid 0,1 for special tokens
            tgt = torch.flip(seq, dims=[0])
            self.data.append((seq, tgt))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        src, tgt = self.data[idx]
        return src, tgt


PAD_ID = 0
BOS_ID = 1  # Begin of sequence
EOS_ID = 2  # End of sequence


# ------------------------- Collate Function ---------------------------- #

def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_batch = [s for s in src_batch]
    tgt_batch = [[BOS_ID] + t.tolist() + [EOS_ID] for t in tgt_batch]
    
    src_lens = [len(s) for s in src_batch]
    tgt_lens = [len(t) for t in tgt_batch]
    
    src_pad = nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=PAD_ID)
    tgt_pad = nn.utils.rnn.pad_sequence([torch.tensor(t) for t in tgt_batch], batch_first=True, padding_value=PAD_ID)
    
    return src_pad, tgt_pad


# -------------------------- Create DataLoader and Model --------------------------- #

dataset = ToyDataset(vocab_size=30, seq_len=6, size=5000)
loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

src_vocab = tgt_vocab = 30
model = TransformerSeq2Seq(src_vocab, tgt_vocab, d_model=256, n_layers=2, n_heads=4).to(device)


# ------------------------- Training Loop --------------------------- #

criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
optimizer = optim.Adam(model.parameters(), lr=3e-4)

EPOCHS = 10

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    
    for src, tgt in loader:
        src, tgt = src.to(device), tgt.to(device)

        # Input to decoder excludes last token (teacher forcing)
        tgt_inp = tgt[:, :-1]       # e.g. [BOS, 4,3,2,1]
        tgt_labels = tgt[:, 1:]     # e.g. [4,3,2,1,EOS]

        # Forward pass
        logits = model(src, tgt_inp)   # (B, T, vocab)

        # Reshape for loss
        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            tgt_labels.reshape(-1)
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    avg_loss = total_loss / len(loader)
    print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {avg_loss:.4f}")


# --------------------- Inference (Greedy Decoding) --------------------- #

def greedy_decode(model, src, max_len=10):
    model.eval()
    src = src.unsqueeze(0).to(device)
    enc_out = model.encoder(src)
    tgt = torch.tensor([[BOS_ID]], device=device)
    
    for _ in range(max_len):
        tgt_mask = model.make_tgt_mask(tgt)
        out = model.decoder(tgt, enc_out, tgt_mask=tgt_mask)
        next_token = out[:, -1, :].argmax(-1).unsqueeze(1)
        tgt = torch.cat([tgt, next_token], dim=1)
        if next_token.item() == EOS_ID:
            break
    return tgt.squeeze(0)


src_seq = torch.tensor([5, 8, 9, 7, 6])
pred = greedy_decode(model, src_seq)
print("SRC:", src_seq.tolist())
print("PRED:", pred.tolist())











# ============================================================
# Test Run
# ============================================================
# if __name__ == "__main__":
#     src_vocab, tgt_vocab = 1000, 1000
#     model = TransformerSeq2Seq(src_vocab, tgt_vocab, d_model=256, n_layers=2, n_heads=4)

#     src = torch.randint(0, src_vocab, (2, 10))   # (batch=2, src_len=10)
#     tgt = torch.randint(0, tgt_vocab, (2, 8))    # (batch=2, tgt_len=8)

#     logits = model(src, tgt)  # (B, T, vocab)
#     print("Output shape:", logits.shape)  # (2, 8, 1000)
