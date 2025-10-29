
# miniBERT with simple tokenizer + MLM training utilities (educational, runnable)
# This cell implements:
# 1) A simple tokenizer built from a local corpus (whitespace + punctuation split)
# 2) get_batch() that yields token-id batches and MLM masked labels (BERT-style masking)
# 3) A miniBERT encoder (encoder-only Transformer) with MLM head (tied to embeddings optional)
# 4) A short sanity run (one forward + backward) to validate shapes and pipeline

# miniBERT_mlm_from_scratch.py
# Minimal educational MiniBERT with tokenizer-from-scratch + MLM pipeline.

import re, math, random, time, os
from collections import Counter
import torch
from torch import nn
import torch.nn.functional as F

# -------------------- USER CONFIG --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# If you want to use your text file set CORPUS_PATH to it; else demo uses `sample_corpus`
CORPUS_PATH = None  # e.g., "data/my_corpus.txt"
sample_corpus = """
Transformers are powerful models for NLP. BERT is an encoder-only Transformer that excels at understanding.
Masked Language Modeling trains a model to predict masked tokens. This demo builds a tiny tokenizer and miniBERT.
Use a larger real corpus for good results: Wikipedia, BookCorpus, or your domain text files.
"""

# Model / training config (small for quick demo; scale up for real training)
VOCAB_SIZE = 2000     # max vocabulary size (including special tokens)
SEQ_LEN = 32          # effective token window length (excluding CLS/SEP)
D_MODEL = 128
N_HEADS = 4
FF_DIM = 256
N_LAYERS = 4
BATCH_SIZE = 32
MLM_PROB = 0.15
LR = 2e-4
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

# -------------------- 1) Simple tokenizer (from scratch) --------------------
class SimpleTokenizer:
    """
    Very simple whitespace + punctuation tokenizer + frequency-based vocab builder.
    NOT a production tokenizer (no subword/BPE) — but fine for learning.
    """
    def __init__(self, vocab_size=VOCAB_SIZE, min_freq=1, special_tokens=None):
        self.vocab_size = vocab_size
        self.min_freq = min_freq
        if special_tokens is None:
            special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        self.special_tokens = special_tokens
        self.token_to_id = {}
        self.id_to_token = {}
        self.vocab_built = False

    def _normalize(self, text):
        text = text.lower()
        text = re.sub(r"([.,!?;:()\"'])", r" \1 ", text)   # space out punctuation
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def tokenize(self, text):
        text = self._normalize(text)
        tokens = text.split(" ")
        return tokens

    def build_vocab(self, corpus_texts):
        cnt = Counter()
        for t in corpus_texts:
            toks = self.tokenize(t)
            cnt.update(toks)
        # keep top (vocab_size - special_tokens)
        max_body = max(0, self.vocab_size - len(self.special_tokens))
        body_tokens = [tok for tok, freq in cnt.most_common() if freq >= self.min_freq][:max_body]
        final_tokens = self.special_tokens + body_tokens
        self.token_to_id = {t: i for i, t in enumerate(final_tokens)}
        self.id_to_token = {i: t for t, i in self.token_to_id.items()}
        self.vocab_built = True
        print(f"[Tokenizer] Built vocab size: {len(final_tokens)} (including {len(self.special_tokens)} special tokens)")

    def encode(self, text, add_special_tokens=True, max_len=None):
        assert self.vocab_built, "Call build_vocab() first"
        toks = self.tokenize(text)
        ids = [self.token_to_id.get(t, self.token_to_id["[UNK]"]) for t in toks]
        if add_special_tokens:
            ids = [self.token_to_id["[CLS]"]] + ids + [self.token_to_id["[SEP]"]]
        if max_len is not None:
            if len(ids) > max_len:
                ids = ids[:max_len]
            else:
                ids = ids + [self.token_to_id["[PAD]"]] * (max_len - len(ids))
        return ids

    def decode(self, ids):
        return " ".join(self.id_to_token.get(i, "[UNK]") for i in ids)

# -------------------- Load corpus texts --------------------
def load_corpus(path=None):
    if path is None:
        return [sample_corpus.strip()]
    assert os.path.exists(path), f"{path} not found"
    texts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                texts.append(line)
    return texts

texts = load_corpus(CORPUS_PATH)
tokenizer = SimpleTokenizer(vocab_size=VOCAB_SIZE)
tokenizer.build_vocab(texts)

# quick inspect
print("Example text:", texts[0][:200])
print("Example tokens:", tokenizer.tokenize(texts[0])[:20])
print("Example encode (first 20 ids):", tokenizer.encode(texts[0], max_len=20)[:20])

# -------------------- 2) Build token stream & get_batch (MLM mask) --------------------
def build_token_stream(texts, tokenizer):
    """
    Concatenate tokenized lines into a long token stream (with SEP tokens between lines).
    We'll sample random windows from this stream to form training batches.
    """
    stream = []
    for t in texts:
        ids = tokenizer.encode(t, add_special_tokens=False)
        stream.extend(ids + [tokenizer.token_to_id["[SEP]"]])
    return stream

token_stream = build_token_stream(texts, tokenizer)
print("Token stream length:", len(token_stream))

def mask_tokens(inputs, tokenizer, mlm_prob=MLM_PROB):
    """
    inputs: LongTensor (batch, seq) containing token ids (with CLS/SEP/PAD)
    Returns:
      masked_inputs: token ids after replacement (some tokens replaced with [MASK] or random)
      labels: -100 for non-masked positions, token_id for masked positions (suitable for CrossEntropy ignore_index=-100)
    """
    inputs = inputs.clone()
    labels = inputs.clone()
    special_ids = {tokenizer.token_to_id[t] for t in ["[PAD]", "[CLS]", "[SEP]"]}

    prob_matrix = torch.full(inputs.shape, mlm_prob)
    for sid in special_ids:
        prob_matrix = prob_matrix.masked_fill(inputs == sid, 0.0)
    mask_mask = torch.bernoulli(prob_matrix).bool()

    labels[~mask_mask] = -100  # ignore loss for non-masked positions

    # Replacement policy
    mask_id = tokenizer.token_to_id["[MASK]"]
    vocab_start = len(tokenizer.special_tokens)
    vocab_end = len(tokenizer.token_to_id) - 1

    rand = torch.rand(inputs.shape)
    for i in range(inputs.size(0)):
        for j in range(inputs.size(1)):
            if not mask_mask[i, j]:
                continue
            r = rand[i, j].item()
            if r < 0.8:
                inputs[i, j] = mask_id
            elif r < 0.9:
                if vocab_end >= vocab_start:
                    inputs[i, j] = random.randint(vocab_start, vocab_end)
            else:
                pass  # keep original

    return inputs, labels

def get_batch(batch_size=BATCH_SIZE, seq_len=SEQ_LEN):
    """
    Sample random windows of seq_len tokens from token_stream.
    Each example returned will be of length seq_len + 2 after adding CLS and SEP:
      [CLS] token_1 token_2 ... token_seq_len [SEP]
    Then apply MLM masking and return masked inputs and labels.
    """
    inputs = torch.full((batch_size, seq_len+2), tokenizer.token_to_id["[PAD]"], dtype=torch.long)
    for i in range(batch_size):
        if len(token_stream) <= seq_len:
            start = 0
        else:
            start = random.randint(0, max(0, len(token_stream) - seq_len))
        window = token_stream[start:start+seq_len]
        ids = [tokenizer.token_to_id["[CLS]"]] + window + [tokenizer.token_to_id["[SEP]"]]
        if len(ids) < seq_len+2:
            ids = ids + [tokenizer.token_to_id["[PAD]"]] * (seq_len+2 - len(ids))
        inputs[i] = torch.tensor(ids[:seq_len+2], dtype=torch.long)

    attention_mask = (inputs != tokenizer.token_to_id["[PAD]"]).long()
    inputs_masked, labels = mask_tokens(inputs, tokenizer)
    return inputs_masked.to(device), labels.to(device), attention_mask.to(device)

# Quick test
xb, yb, attn = get_batch(batch_size=4, seq_len=16)
print("Batch shapes:", xb.shape, yb.shape, attn.shape)
print("Masked input example (ids):", xb[0].tolist())
print("Labels example (ids or -100):", yb[0].tolist())

# -------------------- 3) MiniBERT model (encoder-only) with MLM head --------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)
    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]

def scaled_dot_product_attention(q, k, v, mask=None):
    dk = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(dk)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, v)
    return out

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out = nn.Linear(d_model, d_model)
    def forward(self, x, attn_mask=None):
        b, seq_len, _ = x.size()
        qkv = self.qkv(x).view(b, seq_len, 3, self.n_heads, self.head_dim).permute(2,0,3,1,4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (batch, n_heads, seq_len, head_dim)
        q = q.reshape(b*self.n_heads, seq_len, self.head_dim)
        k = k.reshape(b*self.n_heads, seq_len, self.head_dim)
        v = v.reshape(b*self.n_heads, seq_len, self.head_dim)
        mask = None
        if attn_mask is not None:
            # attn_mask: (batch, seq) -> to (batch*n_heads, seq, seq)
            mask = attn_mask.unsqueeze(1).unsqueeze(1).repeat(1, self.n_heads, seq_len, 1)
            mask = mask.view(b*self.n_heads, seq_len, seq_len)
        out = scaled_dot_product_attention(q, k, v, mask=mask)
        out = out.view(b, self.n_heads, seq_len, self.head_dim).permute(0,2,1,3).contiguous().view(b, seq_len, -1)
        return self.out(out)

class FeedForward(nn.Module):
    def __init__(self, d_model, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, ff_dim, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, attn_mask=None):
        # Pre-LN variant
        x_norm = self.norm1(x)
        attn_out = self.attn(x_norm, attn_mask)
        x = x + self.dropout(attn_out)
        x_norm = self.norm2(x)
        ff_out = self.ff(x_norm)
        x = x + self.dropout(ff_out)
        return x

class MiniBERT(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_layers=4, n_heads=4, ff_dim=256, max_len=512, pad_token_id=0, tie_embeddings=True):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len)
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, ff_dim) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        # MLM head
        self.mlm_head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_embeddings:
            self.mlm_head.weight = self.token_emb.weight
        self.mlm_bias = nn.Parameter(torch.zeros(vocab_size))
        self.pad_token_id = pad_token_id

    def forward(self, input_ids, attention_mask=None):
        x = self.token_emb(input_ids) * math.sqrt(self.d_model)
        x = self.pos_enc(x)
        for layer in self.layers:
            x = layer(x, attn_mask=attention_mask)
        x = self.norm(x)
        logits = self.mlm_head(x) + self.mlm_bias  # (batch, seq, vocab)
        return logits

# instantiate
model = MiniBERT(vocab_size=len(tokenizer.token_to_id),
                 d_model=D_MODEL, n_layers=N_LAYERS, n_heads=N_HEADS, ff_dim=FF_DIM,
                 max_len=SEQ_LEN+2, pad_token_id=tokenizer.token_to_id["[PAD]"]).to(device)
print("Model params (M):", sum(p.numel() for p in model.parameters())/1e6)

# -------------------- 4) Loss, eval, generate helpers --------------------
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

def compute_mlm_loss(logits, labels):
    b, seq, vocab = logits.shape
    return loss_fct(logits.view(-1, vocab), labels.view(-1))

def evaluate(model, n_batches=10):
    model.eval()
    tot_loss = 0.0
    tot_masked = 0
    with torch.no_grad():
        for _ in range(n_batches):
            xb, yb, attn = get_batch(batch_size=BATCH_SIZE, seq_len=SEQ_LEN)
            logits = model(xb, attention_mask=attn)
            loss = compute_mlm_loss(logits, yb)
            tot_loss += loss.item()
            tot_masked += (yb != -100).sum().item()
    model.train()
    print(f"[Eval] avg loss: {tot_loss / max(1, n_batches):.4f}, avg masked tokens per batch: {tot_masked / (n_batches*BATCH_SIZE):.2f}")

def generate_masked_fill(model, text_with_masks, topk=5):
    """
    Simple masked-fill: accept a text string which contains literal token '[MASK]' (space-separated).
    Returns top-k candidate tokens for each mask position.
    """
    model.eval()
    toks = tokenizer.tokenize(text_with_masks)
    ids = []
    for t in toks:
        if t == "[mask]":
            ids.append(tokenizer.token_to_id["[MASK]"])
        else:
            ids.append(tokenizer.token_to_id.get(t, tokenizer.token_to_id["[UNK]"]))
    ids = [tokenizer.token_to_id["[CLS]"]] + ids + [tokenizer.token_to_id["[SEP]"]]
    if len(ids) < SEQ_LEN+2:
        ids = ids + [tokenizer.token_to_id["[PAD]"]] * (SEQ_LEN+2 - len(ids))
    else:
        ids = ids[:SEQ_LEN+2]
    input_ids = torch.tensor([ids], dtype=torch.long).to(device)
    attn_mask = (input_ids != tokenizer.token_to_id["[PAD]"]).long().to(device)
    with torch.no_grad():
        logits = model(input_ids, attention_mask=attn_mask)
        probs = F.softmax(logits, dim=-1)
        mask_positions = (input_ids == tokenizer.token_to_id["[MASK]"]).nonzero(as_tuple=True)
        outputs = []
        for i in range(mask_positions[1].size(0)):
            pos = mask_positions[1][i].item()
            topk_ids = torch.topk(probs[0, pos], k=topk).indices.tolist()
            outputs.append([tokenizer.id_to_token[idx] for idx in topk_ids])
    model.train()
    return outputs

# -------------------- 5) Sanity forward/backward (single step) --------------------
xb, yb, attn = get_batch(batch_size=BATCH_SIZE, seq_len=SEQ_LEN)
logits = model(xb, attention_mask=attn)
print("Logits shape:", logits.shape)
loss = compute_mlm_loss(logits, yb)
print("Initial MLM loss (untrained):", loss.item())

# single backward step (sanity)
loss.backward()
optimizer.step()
optimizer.zero_grad()
print("Sanity single-step done. To train, iterate over many batches and call evaluate() periodically.")

# -------------------- 6) Example training loop (mini) --------------------
# Uncomment and run this loop for actual training (increase STEPS, run on GPU for speed).
#
# EPOCHS = 3
# STEPS_PER_EPOCH = 200
# for epoch in range(EPOCHS):
#     model.train()
#     epoch_loss = 0.0
#     for step in range(STEPS_PER_EPOCH):
#         xb, yb, attn = get_batch(batch_size=BATCH_SIZE, seq_len=SEQ_LEN)
#         logits = model(xb, attention_mask=attn)
#         loss = compute_mlm_loss(logits, yb)
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#         optimizer.step()
#         optimizer.zero_grad()
#         epoch_loss += loss.item()
#     print(f"Epoch {epoch+1} avg loss: {epoch_loss / STEPS_PER_EPOCH:.4f}")
#     evaluate(model, n_batches=20)
#
# After training, try:
# print(generate_masked_fill(model, "transformers are [MASK] models", topk=5))