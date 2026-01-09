import os
import torch
from torch import nn
import timm
from model.decoder import Decoder
import collections
import torch.nn.functional as F
import numpy as np
import cv2
from einops import rearrange  # 导入rearrange函数，用于张量维度重排

def weight_init_backbone(module):
    for n, m in module.named_children():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, (
                nn.ReLU, nn.Sigmoid, nn.PReLU, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool1d, nn.Sigmoid, nn.Identity)):
            pass
        else:
            try:
                m.initialize()
            except:
                pass


def weight_init(module):
    for n, m in module.named_children():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, (
                nn.ReLU, nn.Sigmoid, nn.PReLU, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool1d, nn.Identity)):
            pass
        else:
            try:
                m.initialize()
            except:
                pass


class CamoFormer(torch.nn.Module):
    def __init__(self, cfg, load_path=None):
        super(CamoFormer, self).__init__()
        self.cfg = cfg

        # 1. 初始化Swin-B编码器（禁用自动预训练权重下载）
        self.encoder = timm.create_model(
            'swin_large_patch4_window7_224',  # 匹配图片中的Swin-B模型
            pretrained=False,  # 手动加载本地权重，不自动下载
            features_only=True,
            out_indices=(0, 1, 2, 3)  # 输出4个阶段的特征图
        )

        # 2. 加载本地预训练权重（解决Hugging Face下载问题）
        if cfg is not None and hasattr(cfg, 'swinb_pretrained_path') and os.path.exists(cfg.swinb_pretrained_path):
            pretrained_dict = torch.load(cfg.swinb_pretrained_path, map_location='cpu')
            # 处理权重文件的嵌套格式（如带'model.'或'state_dict'前缀）
            if 'model' in pretrained_dict:
                pretrained_dict = pretrained_dict['model']
            elif 'state_dict' in pretrained_dict:
                pretrained_dict = pretrained_dict['state_dict']
            # 去除键名中的'model.'前缀（若存在）
            pretrained_dict = {k.replace('model.', ''): v for k, v in pretrained_dict.items()}
            # 仅保留编码器中存在且维度匹配的权重
            model_dict = self.encoder.state_dict()
            matched_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
            self.encoder.load_state_dict(matched_dict, strict=False)
            print(f"✅ 成功加载 {len(matched_dict)}/{len(model_dict)} 个Swin-B编码器权重")

        self.decoder = Decoder(128)  # 需确保解码器适配Swin-B的特征通道（128, 256, 512, 1024）
        self.initialize()

    def forward(self, x, shape=None, name=None):
        features = self.encoder(x)
        x1, x2, x3, x4 = features  # x1:128, x2:256, x3:512, x4:1024

        # 修复1：调整特征图传入顺序（高通道→低通道）
        # 修复2：确保所有输入到Decoder的特征图都是CHW格式
        if len(x1.shape) == 4 and x1.shape[1] != 128:  # 若通道维度不在第2位，转置
            x1 = rearrange(x1, 'b h w c -> b c h w')
            x2 = rearrange(x2, 'b h w c -> b c h w')
            x3 = rearrange(x3, 'b h w c -> b c h w')
            x4 = rearrange(x4, 'b h w c -> b c h w')

        if shape is None:
            shape = x.size()[2:]

        # 传入顺序改为：x4(1024)、x3(512)、x2(256)、x1(128)，匹配Decoder的side_conv配置
        P5, P4, P3, P2, P1 = self.decoder(x4, x3, x2, x1, shape)
        return P5, P4, P3, P2, P1

    def initialize(self):
        if self.cfg is not None and getattr(self.cfg, 'snapshot', False):
            self.load_state_dict(torch.load(self.cfg.snapshot))
        else:
            weight_init(self.decoder)