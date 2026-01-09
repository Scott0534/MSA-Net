import os
import torch
from torch import nn
from torchvision import models  # 导入PyTorch官方模型库
from model.decoder import Decoder
import torch.nn.functional as F


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
        elif isinstance(m, (nn.ReLU, nn.Sigmoid, nn.PReLU, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool1d, nn.Identity)):
            pass
        else:
            try:
                m.initialize()
            except:
                pass


class CamoFormer(torch.nn.Module):
    def __init__(self, pretrained=True, local_weight_path=None):
        super(CamoFormer, self).__init__()

        # 1. 加载PyTorch官方ResNet50模型
        # 官方ResNet50结构：conv1→bn1→relu→maxpool→layer1→layer2→layer3→layer4
        # 其中layer1~layer4对应4个特征阶段，通道分别为256、512、1024、2048
        self.resnet = models.resnet34(pretrained=True)  # 先不加载权重，后续手动处理

        # 2. 加载预训练权重（优先本地路径，其次官方预训练）
        if local_weight_path is not None and os.path.exists(local_weight_path):
            # 加载本地权重文件
            pretrained_dict = torch.load(local_weight_path, map_location='cpu')
            # 官方ResNet权重键名无特殊前缀，直接匹配
            model_dict = self.resnet.state_dict()
            matched_dict = {k: v for k, v in pretrained_dict.items() if
                            k in model_dict and v.shape == model_dict[k].shape}
            self.resnet.load_state_dict(matched_dict, strict=False)
            print(f"✅ 成功加载本地ResNet50权重：{len(matched_dict)}/{len(model_dict)} 个参数匹配")
        elif pretrained:
            # 加载PyTorch官方预训练权重（自动下载到缓存目录，如~/.cache/torch/hub/checkpoints/）
            self.resnet = models.resnet50(pretrained=True)
            print("✅ 成功加载PyTorch官方ResNet50预训练权重")
        else:
            print("⚠️ 未加载预训练权重，使用随机初始化")

        # 3. 定义编码器特征提取层（从ResNet中拆分出需要的特征阶段）
        self.encoder = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
            self.resnet.maxpool,
            self.resnet.layer1,  # 特征1：256通道
            self.resnet.layer2,  # 特征2：512通道
            self.resnet.layer3,  # 特征3：1024通道
            self.resnet.layer4  # 特征4：2048通道
        )

        # 4. 解码器适配ResNet50的特征通道
        self.decoder = Decoder(128)  # 与ResNet特征通道（256→512→1024→2048）适配
        self.initialize()

    def forward(self, x, shape=None, name=None):
        # 保存原始输入图像的尺寸（这里的x是未经过任何下采样的输入）
        original_shape = x.size()[2:]  # 正确：原始输入尺寸（如224x224）

        # 提取4个阶段的特征（后续x会被下采样，但不影响original_shape）
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x1 = self.resnet.layer1(x)  # 256通道
        x2 = self.resnet.layer2(x1)  # 512通道
        x3 = self.resnet.layer3(x2)  # 1024通道
        x4 = self.resnet.layer4(x3)  # 2048通道

        if shape is None:
            shape = original_shape  # 使用原始输入尺寸，而非下采样后的尺寸

        # 解码器上采样到原始输入尺寸
        P5, P4, P3, P2, P1 = self.decoder(x4, x3, x2, x1, shape)
        return P5, P4, P3, P2, P1

    def initialize(self):
        # 初始化解码器权重
        weight_init(self.decoder)