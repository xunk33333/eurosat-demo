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
        self.patch_embed = PatchEmbedding(config)
        self.encoder = nn.Sequential(*[
            TransformerEncoder(config) for _ in range(config.vit_num_layers)
        ])
        self.classifier = nn.Linear(config.vit_embed_dim, config.num_classes)
        
    def forward(self, x):
        x = self.patch_embed(x)
        x = self.encoder(x)
        return self.classifier(x[:, 0])

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
        block = BasicBlock
        layers = {
            'resnet18': [2, 2, 2, 2],
            'resnet34': [3, 4, 6, 3],
            'resnet50': [3, 4, 6, 3],
            'resnet101': [3, 4, 23, 3],
            'resnet152': [3, 8, 36, 3]
        }[config.resnet_version]
        
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, config.num_classes)
        
    def _make_layer(self, block, out_channels, blocks, stride):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x

def get_model(config: Config):
    if config.model_type == 'vit':
        return ViTClassifier(config)
    elif config.model_type == 'resnet':
        return ResNetClassifier(config)
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")
