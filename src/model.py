"""Vision Transformer for species classification (num_classes is passed in by
the caller, e.g. 22 for the full dataset; NUM_CLASSES below is just a fallback
default used by the __main__ smoke test).

The model exposes:
    forward(x)          -> class logits, shape (B, num_classes)
    get_embeddings(x)   -> pooled embeddings (CLS token), shape (B, embed_dim)

get_embeddings() is what scripts/evaluate.py uses for the PCA / t-SNE step.
"""

import torch
import torch.nn as nn
from torchvision.models.vision_transformer import VisionTransformer

NUM_CLASSES = 5


class ClassificationHead(nn.Module):
    """head: LayerNorm -> Dropout -> Linear."""

    def __init__(self, embed_dim: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_dim, num_classes)

        # Matches the original ViT paper's head init convention: small
        # truncated-normal weights, zero bias, rather than PyTorch's default.
        nn.init.trunc_normal_(self.fc.weight, std=0.02)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.dropout(self.norm(x)))


class SpeciesViT(nn.Module):
    """ViT trained from scratch for species classification.

    Defaults give a small ViT (~11M params at 224px with patch 16), which is
    a sensible size for training from random weights on a modest dataset.
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        image_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 384,
        depth: int = 6,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
    ):
        super().__init__()
        self.config = dict(
            num_classes=num_classes,
            image_size=image_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            attention_dropout=attention_dropout,
        )

        # Random-weight backbone (torchvision initialises randomly by default).
        # image_size/patch_size fix the patch grid (224/16 = 14x14 = 196
        # patches); hidden_dim is the embedding each patch is linearly
        # projected into; mlp_dim is the transformer MLP's expanded width.
        self.backbone = VisionTransformer(
            image_size=image_size,
            patch_size=patch_size,
            num_layers=depth,
            num_heads=num_heads,
            hidden_dim=embed_dim,
            mlp_dim=int(embed_dim * mlp_ratio),
            dropout=dropout,
            attention_dropout=attention_dropout,
            num_classes=num_classes,  # replaced below
        )
        # Remove torchvision's built-in head so backbone(x) returns the pooled
        # CLS embedding (B, embed_dim). We attach our own head instead.
        self.backbone.heads = nn.Identity()

        self.head = ClassificationHead(embed_dim, num_classes, dropout)

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Pooled CLS-token embedding, shape (B, embed_dim)."""
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.get_embeddings(x))


def build_model(num_classes: int = NUM_CLASSES, **kwargs) -> SpeciesViT:
    return SpeciesViT(num_classes=num_classes, **kwargs)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick smoke test: python -m src.model  
    model = build_model()
    x = torch.randn(2, 3, 224, 224)
    logits = model(x)
    emb = model.get_embeddings(x)
    print(f"logits: {tuple(logits.shape)}  embeddings: {tuple(emb.shape)}")
    print(f"trainable parameters: {count_parameters(model):,}")