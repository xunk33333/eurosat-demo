from dataclasses import dataclass

import torch

@dataclass
class Config:
    # Dataset configuration
    data_root: str = "./data/EuroSAT"
    img_size: int = 224
    batch_size: int = 32
    num_workers: int = 4
    
    # Model configuration
    model_type: str = "resnet"  # "vit" or "resnet"
    num_classes: int = 10
    
    # ViT parameters
    vit_embed_dim: int = 768
    vit_num_heads: int = 12
    vit_num_layers: int = 12
    vit_patch_size: int = 16
    vit_pretrained: bool = True
    
    # ResNet parameters
    resnet_version: str = "resnet18"  # resnet18/34/50/101/152
    resnet_pretrained: bool = True
    
    # Training parameters
    learning_rate: float = 1e-2
    num_epochs: int = 100
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
