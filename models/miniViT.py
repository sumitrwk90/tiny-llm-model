
import torch
import torch.nn as nn

class MiniViT(nn.Module):
    def __init__(self, image_size=32, patch_size=8, num_classes=10, dim=128, depth=4, heads=4, mlp_dim=256):
        super().__init__()
        num_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size * patch_size

        # Patch embedding
        self.to_patch_emb = nn.Sequential(
            nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size),
            nn.Flatten(2),
            nn.Transpose(1, 2)
        )

        # CLS token & positional embeddings
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_emb = nn.Parameter(torch.randn(1, num_patches + 1, dim))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # Classification head
        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

    def forward(self, img):
        x = self.to_patch_emb(img)  # [B, num_patches, dim]
        b, n, _ = x.shape
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_emb
        x = self.transformer(x)
        return self.mlp_head(x[:, 0])  # CLS token output
