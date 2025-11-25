
# -------------------------------- Overfitting Test -------------------------------- #
0.52 / 68.75



from msilib.schema import _Validation_records
import torch
from torch import nn
import torch.nn.functional as F
import tiktoken
import matplotlib.pyplot as plt

# from Activation_Function import GELU
# from src.TransformerBlock import TransformerBlock
# from src.Normalization_Layer import LayerNorm


class MLP_LM(nn.Module):
    def __init__(self, vocab_size, emb_dim, dropout=0.0):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, emb_dim)
        self.mlp = nn.Sequential(
                        nn.Linear(emb_dim, 4*emb_dim),
                        nn.GELU(),
                        nn.Linear(4*emb_dim, emb_dim),
                        nn.GELU(),
                        nn.Linear(emb_dim, emb_dim),
                        nn.Dropout(dropout)
        )
        self.out = nn.Linear(emb_dim, vocab_size)

    def forward(self, idx):
        b, t = idx.shape
        x = self.tok_emb(idx)
        
        x = self.mlp(x)
        x = x.view(b*t, -1)
        logits = self.out(x)
        
        
        return logits
    

# ------------------- Read Data ------------------- #

with open("/tiny-llm-model/data_source/the-verdict.txt", "r", encoding="utf-8") as f:
    text = f.read()

# keep only the first 1000 characters
text = text[:1000]

# build vocabulary
chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}

def encode(s): return [stoi[c] for c in s]
def decode(ids): return ''.join([itos[i] for i in ids])

data = torch.tensor(encode(text), dtype=torch.long)
print("Dataset tokens:", len(data), "Vocab size:", vocab_size)


# tokenizer = tiktoken.get_encoding("gpt2")


# def get_batch(data, batch_size, seq_len):
#     idx = torch.randint(0, data.size(0)-seq_len, (batch_size,))
#     x = torch.stack([data[i:i+seq_len] for i in idx])
#     y = torch.stack([data[i+1:i+seq_len+1] for i in idx])
#     return x, y

def get_batch(data, block_size, start_idx=0):
    x = data[start_idx:start_idx+block_size].unsqueeze(0)
    y = data[start_idx+1:start_idx+block_size+1].unsqueeze(0)
    return x, y



batch_size=8
seq_len=32

# x, y = get_batch(data, batch_size=8, seq_len=32)
# print(f"x----> {x}")
# print(f"y----> {y}")
    
emb_dim=256

model = MLP_LM(vocab_size=vocab_size, emb_dim=256)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# ----------------- Trining Loop ----------------- #

for step in range(10000):
    xb, yb = get_batch(data, block_size=32)

    logits = model(xb)
    loss = F.cross_entropy(logits, yb.view(-1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        print(f"Step {step:4d} | Loss: {loss.item():.4f}")


@torch.no_grad()
def evaluate_accuracy(model, data):
    xb, yb = get_batch(data, block_size=32)
    logits = model(xb)
    preds = torch.argmax(logits, dim=-1)
    acc = (preds == yb.view(-1)).float().mean().item()
    return acc

acc = evaluate_accuracy(model, data)
print(f"Training accuracy: {acc * 100:.2f}%")


# # --- Step 6. Plot loss curve ---
# plt.figure(figsize=(6,4))
# plt.plot(loss, label="Training loss")
# plt.xlabel("Step")
# plt.ylabel("CrossEntropy Loss")
# plt.title("Tiny MLP LM Memorization Curve")
# plt.legend()
# plt.grid(True)
# plt.show()