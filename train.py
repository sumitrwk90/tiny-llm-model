import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import random
import os
import numpy as np

import tiktoken
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from models.model_Anima_350M import Anima

# ----------- CUDA environment set-up ----------- #
device = "cuda" if torch.cuda.is_available() else "cpu"
model = Anima()
model.to(device)
# ---------------- Initialize distributed process group ---------------- #
def setup_distributed():
    # These are automatically set by torchrun
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    dist.init_process_group(
        backend="nccl",       # GPU backend (use "gloo" for CPU-only)
        rank=rank,
        world_size=world_size
    )
    torch.cuda.set_device(local_rank)

    return local_rank, rank, world_size

# ---------------- Helper: Check if this is main process ----------------#
def is_main_process(local_rank):

    return local_rank == 0

if device=="cuda":
    local_rank, rank, world_size = setup_distributed()
    # ------------DDP model wraper ------------- #
    model.to(local_rank)
    model = DDP(model, device_ids=[local_rank])

# ------------- Optimizer ------------- #

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)  # weight_decay=0.1, (decay_factor)betas = (0.9, 0.95), learning_rate=3e-4

# ------------ Optional: LR scheduler ------------ #

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2000)    # optimizer=optimizer, learning_rate=3e-4, learning_rate_iter= , final_learning_rate= ,

# ------------- Optional: Mixed precision scaler ------------- #

scaler = torch.cuda.amp.GradScaler()

# -------------- Resume Training -------------- #

save_dir = os.path.join(OUT_DIR if "OUT_DIR" in globals() else "/content")
os.makedirs(save_dir, exist_ok=True)
resume_path = os.path.join(OUT_DIR, "ckpt.pt")
# resume_path = "ckpt.pt"
start_step = 0

if os.path.exists(resume_path):
    print(f"Resuming from checkpoint: {resume_path}")
    checkpoint = torch.load(resume_path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    scheduler.load_state_dict(checkpoint["scheduler"])
    scaler.load_state_dict(checkpoint["scaler"])

    start_step = checkpoint["step"] + 1  # continue from next step

    # Restore RNG states for exact reproducibility
    # if "rng_state" in checkpoint:
    #     torch.set_rng_state(checkpoint["rng_state"])
    state = checkpoint["rng_state"]
    if isinstance(state, list):
        restored = [torch.ByteTensor(s) for s in state]
        torch.set_rng_state(restored)

    # if "cuda_rng_state" in checkpoint and torch.cuda.is_available():
    #     torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
    state = checkpoint["cuda_rng_state"]
    if isinstance(state, list):
    # convert list-of-lists back to ByteTensors
        restored = [s.to('cpu') for s in state]
        torch.cuda.set_rng_state_all(restored)

    if "numpy_rng_state" in checkpoint:
        np.random.set_state(checkpoint["numpy_rng_state"])

    if "python_rng_state" in checkpoint:
        random.setstate(checkpoint["python_rng_state"])


# -------------- Training Loop --------------- #

max_step=100001
# stop_at_step = 800
save_interval = 500

for step in range(start_step, max_step):
    xb, yb = get_batch("train")
    """
    xb = [batch, block_size]     block_size = context_len
    """
    xb, yb = xb.to(device), yb.to(device)

    # ---- Model gives Output ---->
    logits = model(xb)

    # ---- Loss Function ---->
    loss = F.cross_entropy(logits.view(-1, vocab_size), yb.view(-1))
    """
    cross-Entropy = -∑ y*log(y_pred)
    """
    # ---- Back Tracking ---->
    optimizer.zero_grad()
    loss.backward()

    # 1.Gradient clipping (prevent expolsion)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # max_norm = grad_clip = 1.0

    # 2. Optimizer with AdamW (adaptive updates + weight decay)
    optimizer.step()
    # 3. Learning rate scheduler
    scheduler.step()

    # <|---------Save checkpoint every N-step---------|>
    if step ==  max_step-1 or step % save_interval == 0:   #step == max_step-1 or #

        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate()
        }

        torch.save(checkpoint, resume_path)
        print(f"Checkpoint saved at step: {step}")
        # break

    # Logging
    if step % 500 == 0:
        print(f"Step :{step}, loss :{loss.item():.4f}")

# dist.destroy_process_group()    # Destroy all the ddp configuration, so to no error for the cumulative training or fine-tuning...