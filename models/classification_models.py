import torch.nn as nn
import torch.nn.functional as F
import timm

class ResNet18ClassModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.backbone = timm.create_model('resnet18', pretrained=args.pretrained)

        self.backbone.fc = nn.Linear(in_features=512, out_features=8, bias=True)

    def forward(self, x):
        class_pred = self.backbone(x)
        
        return class_pred

class SwinClassModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.backbone = timm.create_model('swin_small_patch4_window7_224', pretrained=args.pretrained)

        self.backbone.head = nn.Linear(in_features=768, out_features=8, bias=True)

    def forward(self, x):
        class_pred = self.backbone(x)
        
        return class_pred

class ConvNextClassModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.backbone = timm.create_model('convnext_small', pretrained=args.pretrained)
        self.backbone.head.fc = nn.Linear(in_features=768, out_features=8, bias=True)

    def forward(self, x):
        class_pred = self.backbone(x)
        
        return class_pred