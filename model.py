import torch
import torch.nn as nn
import math
from config import Config

class PatchEmbedding(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.patch_size = config.vit_patch_size
        in_channels = 3
        patch_dim = in_channels * self.patch_size ** 2
        
        self.proj = nn.Conv2d(in_channels, config.vit_embed_dim, 
                             kernel_size=self.patch_size, 
                             stride=self.patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.vit_embed_dim))
        self.position_emb = nn.Parameter(torch.randn(1, (config.img_size // self.patch_size) ** 2 + 1, config.vit_embed_dim))
        
    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)  # [B, E, H/P, W/P]
        x = x.flatten(2).transpose(1, 2)  # [B, N, E]
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.position_emb
        
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.attention = nn.MultiheadAttention(config.vit_embed_dim, config.vit_num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(config.vit_embed_dim, 4 * config.vit_embed_dim),
            nn.GELU(),
            nn.Linear(4 * config.vit_embed_dim, config.vit_embed_dim),
        )
        self.norm1 = nn.LayerNorm(config.vit_embed_dim)
        self.norm2 = nn.LayerNorm(config.vit_embed_dim)
        
    def forward(self, x):
        x = self.norm1(x)
        attn_out, _ = self.attention(x, x, x)
        x = x + attn_out
        
        x = self.norm2(x)
        mlp_out = self.mlp(x)
        x = x + mlp_out
        
        return x

class ViTClassifier(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        import torchvision.models as models
        
        # Create torchvision ViT model
        self.model = models.vit_b_16(pretrained=config.vit_pretrained)
        
        # Replace final classification head
        in_features = self.model.heads.head.in_features
        self.model.heads.head = nn.Linear(in_features, config.num_classes)
        
    def forward(self, x):
        return self.model(x)

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.relu(out)
        
        return out

class ResNetClassifier(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        import torchvision.models as models
        
        # Create torchvision resnet model
        model_fn = {
            'resnet18': models.resnet18,
            'resnet34': models.resnet34,
            'resnet50': models.resnet50,
            'resnet101': models.resnet101,
            'resnet152': models.resnet152
        }[config.resnet_version]
        
        self.model = model_fn(pretrained=config.resnet_pretrained)
        
        # Replace final fully connected layer
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, config.num_classes)
        
    def forward(self, x):
        return self.model(x)

def get_model(config: Config):
    if config.model_type == 'vit':
        return ViTClassifier(config)
    elif config.model_type == 'resnet':
        return ResNetClassifier(config)
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")
