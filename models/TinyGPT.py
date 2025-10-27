
import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken



device = torch.device("cpu")
# ------------------ data loading ------------------ #

with open('/tiny-llm-model/data_source/the-verdict.txt', 'r', encoding='utf-8') as f:
    data = f.read()


tokenizer = tiktoken.get_encoding("gpt2")
tokens = tokenizer.encode(data)

n = len(tokens)

train_ids = tokens[:int(n*0.9)]
val_ids = tokens[int(n*0.9):]

# ---------------- Make Batch ---------------- #

def get_batch(data, batch_size, seq_len):
    split_data = train_ids if data == "train" else val_ids
    idx = torch.randint(0, len(split_data)-seq_len, (batch_size,))
    x = torch.stack([split_data[i:i+seq_len] for i in idx]).to(device)
    y = torch.stack([split_data[i+1:i+seq_len+1] for i in idx]).to(device)
    return x, y

x, y = get_batch(train_ids, batch_size=8, seq_len=32)
print(x.shape)