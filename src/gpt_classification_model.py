
import torch
from GPT_Architecture import GPTModel
from gpt_download3 import download_and_load_gpt2
from data_loader import train_dataset, train_loader, test_loader, val_loader
import time
import numpy as np

torch.manual_seed(123)

num_classes = 2

CHOOSE_MODEL = "gpt2-small (124M)"
INPUT_PROMPT = "Every effort moves"

BASE_CONFIG = {
    "vocab_size": 50257,     # Vocabulary size
    "context_length": 1024,  # Context length
    "drop_rate": 0.0,        # Dropout rate
    "qkv_bias": True         # Query-key-value bias
}

model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}


BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

assert train_dataset.max_length <= BASE_CONFIG["context_length"], (
    f"Dataset length {train_dataset.max_length} exceeds model's context "
    f"length {BASE_CONFIG['context_length']}. Reinitialize data sets with "
    f"`max_length={BASE_CONFIG['context_length']}`"
)


# def assign(left, right):
#     if left.shape != right.shape:
#         raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
#     return torch.nn.Parameter(torch.tensor(right))


# # load gpt-2 params in GPTModel
# def load_weights_into_gpt(gpt, params):
#     gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])
#     gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])
    
#     for b in range(len(params["blocks"])):
#         q_w, k_w, v_w = np.split(
#             (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
#         gpt.trf_blocks[b].att.W_query.weight = assign(
#             gpt.trf_blocks[b].att.W_query.weight, q_w.T)
#         gpt.trf_blocks[b].att.W_key.weight = assign(
#             gpt.trf_blocks[b].att.W_key.weight, k_w.T)
#         gpt.trf_blocks[b].att.W_value.weight = assign(
#             gpt.trf_blocks[b].att.W_value.weight, v_w.T)

#         q_b, k_b, v_b = np.split(
#             (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
#         gpt.trf_blocks[b].att.W_query.bias = assign(
#             gpt.trf_blocks[b].att.W_query.bias, q_b)
#         gpt.trf_blocks[b].att.W_key.bias = assign(
#             gpt.trf_blocks[b].att.W_key.bias, k_b)
#         gpt.trf_blocks[b].att.W_value.bias = assign(
#             gpt.trf_blocks[b].att.W_value.bias, v_b)

#         gpt.trf_blocks[b].att.out_proj.weight = assign(
#             gpt.trf_blocks[b].att.out_proj.weight, 
#             params["blocks"][b]["attn"]["c_proj"]["w"].T)
#         gpt.trf_blocks[b].att.out_proj.bias = assign(
#             gpt.trf_blocks[b].att.out_proj.bias, 
#             params["blocks"][b]["attn"]["c_proj"]["b"])

#         gpt.trf_blocks[b].ff.layers[0].weight = assign(
#             gpt.trf_blocks[b].ff.layers[0].weight, 
#             params["blocks"][b]["mlp"]["c_fc"]["w"].T)
#         gpt.trf_blocks[b].ff.layers[0].bias = assign(
#             gpt.trf_blocks[b].ff.layers[0].bias, 
#             params["blocks"][b]["mlp"]["c_fc"]["b"])
#         gpt.trf_blocks[b].ff.layers[2].weight = assign(
#             gpt.trf_blocks[b].ff.layers[2].weight, 
#             params["blocks"][b]["mlp"]["c_proj"]["w"].T)
#         gpt.trf_blocks[b].ff.layers[2].bias = assign(
#             gpt.trf_blocks[b].ff.layers[2].bias, 
#             params["blocks"][b]["mlp"]["c_proj"]["b"])

#         gpt.trf_blocks[b].norm1.scale = assign(
#             gpt.trf_blocks[b].norm1.scale, 
#             params["blocks"][b]["ln_1"]["g"])
#         gpt.trf_blocks[b].norm1.shift = assign(
#             gpt.trf_blocks[b].norm1.shift, 
#             params["blocks"][b]["ln_1"]["b"])
#         gpt.trf_blocks[b].norm2.scale = assign(
#             gpt.trf_blocks[b].norm2.scale, 
#             params["blocks"][b]["ln_2"]["g"])
#         gpt.trf_blocks[b].norm2.shift = assign(
#             gpt.trf_blocks[b].norm2.shift, 
#             params["blocks"][b]["ln_2"]["b"])

#     gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
#     gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
#     gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])


# model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")


# settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")

model = GPTModel(BASE_CONFIG)
model.load_state_dict(torch.load("model.pth"))


# freeze the model...
for param in model.parameters():
    param.requires_grad = False

# Change num head of classification model...
model.out_head = torch.nn.Linear(in_features=BASE_CONFIG["emb_dim"], out_features=num_classes)


# Config last transformer block
for param in model.trf_blocks[-1].parameters():
    param.requires_grad = True

for param in model.final_norm.parameters():
    param.requires_grad = True


def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    model.eval()
    correct_predictions, num_examples = 0, 0

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)

            with torch.no_grad():
                logits = model(input_batch)[:, -1, :]  # Logits of last output token
            predicted_labels = torch.argmax(logits, dim=-1)

            num_examples += predicted_labels.shape[0]
            correct_predictions += (predicted_labels == target_batch).sum().item()
        else:
            break
    return correct_predictions / num_examples


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)[:, -1, :]  # Logits of last output token
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss


# Same as in chapter 5
def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        # Reduce the number of batches to match the total number of batches in the data loader
        # if num_batches exceeds the number of batches in the data loader
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


# Same as chapter 5
def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


# Overall the same as `train_model_simple` in chapter 5
def train_classifier_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                            eval_freq, eval_iter):
    # Initialize lists to track losses and examples seen
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1

    # Main training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad() # Reset loss gradients from previous batch iteration
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward() # Calculate loss gradients
            optimizer.step() # Update model weights using loss gradients
            examples_seen += input_batch.shape[0] # New: track examples instead of tokens
            global_step += 1

            # Optional evaluation step
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # Calculate accuracy after each epoch
        train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=eval_iter)
        val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=eval_iter)
        print(f"Training accuracy: {train_accuracy*100:.2f}% | ", end="")
        print(f"Validation accuracy: {val_accuracy*100:.2f}%")
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)

    return train_losses, val_losses, train_accs, val_accs, examples_seen


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


start_time = time.time()

torch.manual_seed(123)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)

num_epochs = 5
train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=50, eval_iter=5,
)

# Save Model
torch.save(model.state_dict(), "review_classifier.pth")

end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"Training completed in {execution_time_minutes:.2f} minutes.")


train_accuracy = calc_accuracy_loader(train_loader, model, device)
val_accuracy = calc_accuracy_loader(val_loader, model, device)
test_accuracy = calc_accuracy_loader(test_loader, model, device)

print(f"Training accuracy: {train_accuracy*100:.2f}%")
print(f"Validation accuracy: {val_accuracy*100:.2f}%")
print(f"Test accuracy: {test_accuracy*100:.2f}%")