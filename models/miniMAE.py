import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class PatchEmbed(nn.Module):
    def __init__(self, patch_size=8, embed_dim=128):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # (B, C, H, W) → (B, num_patches, embed_dim)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x
        

class MiniMAE(nn.Module):
    def __init__(self, img_size=32, patch_size=8, embed_dim=128, encoder_depth=4, decoder_dim=64):
        super().__init__()
        self.patch_embed = PatchEmbed(patch_size, embed_dim)
        num_patches = (img_size // patch_size) ** 2

        # Positional embedding and mask token
        self.pos_emb = nn.Parameter(torch.randn(1, num_patches, embed_dim))
        self.mask_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # Encoder (ViT-style)
        enc_layer = nn.TransformerEncoderLayer(embed_dim, nhead=4, dim_feedforward=256, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=encoder_depth)

        # Lightweight decoder
        dec_layer = nn.TransformerEncoderLayer(decoder_dim, nhead=4, dim_feedforward=128, batch_first=True)
        self.decoder = nn.TransformerEncoder(dec_layer, num_layers=2)
        self.decoder_proj = nn.Linear(embed_dim, decoder_dim)
        self.reconstruction = nn.Linear(decoder_dim, patch_size * patch_size * 3)

    def forward(self, imgs, mask_ratio=0.75):
        patches = self.patch_embed(imgs)
        B, N, D = patches.shape

        # Create random mask
        num_mask = int(N * mask_ratio)
        noise = torch.rand(B, N, device=imgs.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, num_mask:]
        visible = torch.gather(patches, 1, ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # Encode visible patches
        x = visible + self.pos_emb[:, :visible.shape[1], :]
        latent = self.encoder(x)

        # Decode: combine latent + mask tokens
        mask_tokens = self.mask_token.repeat(B, num_mask, 1)
        x_ = torch.cat([latent, mask_tokens], dim=1)
        x_ = torch.gather(x_, 1, ids_restore.unsqueeze(-1).repeat(1, 1, D))
        x_ = x_ + self.pos_emb

        # Decode and reconstruct
        x_ = self.decoder_proj(x_)
        x_ = self.decoder(x_)
        rec_patches = self.reconstruction(x_)
        return rec_patches, patches, ids_restore, num_mask


def mae_loss(pred, target, ids_restore, num_mask):
    B, N, _ = pred.shape
    mask = torch.zeros(B, N, device=pred.device)
    mask[:, :num_mask] = 1
    mask = torch.gather(mask, 1, ids_restore)
    loss = ((pred - target) ** 2 * mask.unsqueeze(-1)).sum() / mask.sum()
    return loss


# ------------------------- Data Loader -------------------------- #

transform = T.Compose([
    T.Resize((32, 32)),
    T.ToTensor(),
])

trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)


# --------------------- Training Loop ---------------------- #

model = MiniMAE().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(1):
    for imgs, _ in trainloader:
        imgs = imgs.to(device)
        rec, target, ids_restore, num_mask = model(imgs)
        loss = mae_loss(rec, target, ids_restore, num_mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch done | Loss: {loss.item():.4f}")
