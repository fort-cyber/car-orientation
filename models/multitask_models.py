import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNet18MultiModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.backbone = timm.create_model('resnet18', pretrained=args.pretrained)

        self.backbone.fc = nn.Linear(in_features=512, out_features=1024, bias=True)
        
        in_features = 1024
        num_classes = 8

        self.class_prediction = nn.Linear(in_features, num_classes)
        self.angle_prediction = nn.Linear(in_features + num_classes, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.backbone(x)

        class_pred = self.class_prediction(x)
        x = torch.cat((x, class_pred), dim=1)
        angle_pred = self.angle_prediction(x)
        angle_pred = self.sigmoid(angle_pred)

        return class_pred, angle_pred

class ConvNextSmallMultiModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.backbone = timm.create_model('convnext_small', pretrained=args.pretrained)

        self.backbone.head.fc = nn.Linear(in_features=768, out_features=1024, bias=True)
              
        in_features = 1024
        num_classes = 8

        self.class_prediction = nn.Linear(in_features, num_classes)
        self.angle_prediction = nn.Linear(in_features + num_classes, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.backbone(x)

        class_pred = self.class_prediction(x)
        x = torch.cat((x, class_pred), dim=1)
        angle_pred = self.angle_prediction(x)

        angle_pred = self.sigmoid(angle_pred)
        
        return class_pred, angle_pred

class SwinSmallMultiModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.backbone = timm.create_model('swin_small_patch4_window7_224', pretrained=args.pretrained)

        self.backbone.head = nn.Linear(in_features=768, out_features=1024, bias=True)
              
        in_features = 1024
        num_classes = 8

        self.class_prediction = nn.Linear(in_features, num_classes)
        self.angle_prediction = nn.Linear(in_features + num_classes, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.backbone(x)

        class_pred = self.class_prediction(x)
        x = torch.cat((x, class_pred), dim=1)
        angle_pred = self.angle_prediction(x)
        angle_pred = self.sigmoid(angle_pred)

        return class_pred, angle_pred