import os
import torch
import torchvision.transforms as transforms
from datasets import *
from models import *
from augmentation import *
from trainers import *
from evaluators import *
import numpy as np
import random

def get_loaders(args):
    if args.dataset == 'angle':
        train_path = os.path.join(args.dataset_path, 'train.csv')
        test_path = os.path.join(args.dataset_path, 'test.csv')

        train_dataset=AngleDataset(args, train_path, transform=get_augmentation(args))
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True)
        
        test_dataset= AngleDataset(args, test_path, transform=transforms.ToTensor())
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True)
        
    return train_loader, test_loader

def get_scheduler(optimizer, args, steps):
    if args.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.95)
    elif args.scheduler == 'onecycle':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=steps, epochs=args.epochs)
    
    return scheduler

def get_model(args):

    # Regression Models
    if args.model == 'convnext-small':
        model = ConvNextSmallModel(args)
    elif args.model == 'convnext-small-norm':
        model = ConvNextSmallNormModel(args)
    elif args.model == 'resnet18-norm':
        model = ResNet18NormModel(args)
    elif args.model == 'swin-small-norm':
        model = SwinSmallNormModel(args)
    
    # Classification Models
    elif args.model == 'convnext-class':
        model = ConvNextClassModel(args)
    elif args.model == 'resnet18-class':
        model = ResNet18ClassModel(args)
    elif args.model == 'swin-class':
        model = SwinClassModel(args)

    # Multi Models
    elif args.model == 'convnext-multi-norm':
        model = ConvNextSmallMultiModel(args)
    elif args.model == 'resnet18-multi-norm':
        model = ResNet18MultiModel(args)
    elif args.model == 'swin-multi-norm':
        model = SwinSmallMultiModel(args)

    return model

def get_trainer(args, train_loader, device, optimizer, scaler, scheduler, criterion):
    if args.trainer == 'regression':
        return RegressionTrainer(args, train_loader, device, optimizer, scaler, scheduler, criterion)
    elif args.trainer == 'classification':
        return ClassTrainer(args, train_loader, device, optimizer, scaler, scheduler, criterion)
    elif args.trainer == 'multi':
        return MultiTrainer(args, train_loader, device, optimizer, scaler, scheduler, criterion)

def get_criterion(args):
    if args.trainer == 'regression':
        return CircularMeanAbsoluteError()
    elif args.trainer == 'classification':
        return torch.nn.CrossEntropyLoss()
    elif args.trainer == 'multi':
        class_criterion = torch.nn.CrossEntropyLoss()
        regression_criterion = CircularMeanAbsoluteError()
        return (class_criterion, regression_criterion)

def get_evaluator(args, test_loader, device, criterion, run_name, metric):
    if args.trainer == 'regression':
        return RegressionEvaluator(args, test_loader, device, criterion, run_name, metric)
    elif args.trainer == 'classification':
        return ClassEvaluator(args, test_loader, device, criterion, run_name, metric)
    elif args.trainer == 'multi':
        return MultiEvaluator(args, test_loader, device, criterion, run_name, metric)

def get_metric(args):
    if args.trainer == 'regression':
        return CircularDistance()
    elif args.trainer == 'classification':
        return Accuracy()

def get_augmentation(args):
    if args.augmentation == 'no-aug':
        return transforms.Compose([transforms.ToTensor()])
    elif args.augmentation == 'weak-aug':
        return transforms.Compose([transforms.RandomResizedCrop(size=224, scale=(0.8, 1)), transforms.ColorJitter(0.3, 0.3, 0.3, 0.3), transforms.ToTensor()])
    elif args.augmentation == 'rand-aug':
        return transforms.Compose([
				RandAugmentPC(n=2, m=args.magnitude),
				transforms.ToTensor(),
			])

def circular_mean_absolute_error(y_true, y_pred):
    # Convert angles to radians
    y_true_rad = torch.deg2rad(y_true)
    y_pred_rad = torch.deg2rad(y_pred)

    # Calculate the circular difference
    diff = torch.atan2(torch.sin(y_true_rad - y_pred_rad), torch.cos(y_true_rad - y_pred_rad))

    # Convert back to degrees and take the absolute value
    diff_deg = torch.rad2deg(diff)
    abs_diff_deg = torch.abs(diff_deg)

    return abs_diff_deg

def log_images(args, wandb, model, test_loader, device, run_name, epoch):
    model.eval()
    # Get a batch of random images
    dataiter = iter(test_loader)
    data = dataiter.next()
    images = data['img'][:24].to(device)
    labels = data['label'][:24].to(device)

    with torch.no_grad():
        preds = model(images)

    if 'norm' in args.model:
        # If outputs are normalized, multiply by 360 to get degrees
        preds = preds * 360
    else:
        # If outputs are negative, add 360 to make them positive
        preds[preds < 0] += 360

    # Round predictions and labels to 2 decimal places
    preds = torch.round(preds * 100) / 100
    labels = torch.round(labels * 100) / 100
    wandb_images = [wandb.Image(img, caption=f"Prediction: {pred.item()},\n Label: {label}") for img, pred, label in zip(images, preds, labels)]
    wandb.log({"images": wandb_images}, step=epoch)

class CircularMeanAbsoluteError(torch.nn.Module):
    def __init__(self):
        super(CircularMeanAbsoluteError, self).__init__()

    def forward(self, y_true, y_pred):
        # Convert angles to radians
        y_true_rad = torch.deg2rad(y_true)
        y_pred_rad = torch.deg2rad(y_pred)

        # Calculate the circular difference
        diff = torch.atan2(torch.sin(y_true_rad - y_pred_rad), torch.cos(y_true_rad - y_pred_rad))

        # Convert back to degrees and take the absolute value
        diff_deg = torch.rad2deg(diff)
        abs_diff_deg = torch.abs(diff_deg)

        return torch.mean(abs_diff_deg)

class CircularDistance(torch.nn.Module):
    def __init__(self):
        super(CircularDistance, self).__init__()

    def forward(self, y_true, y_pred):
        # Convert angles to radians
        y_true_rad = torch.deg2rad(y_true)
        y_pred_rad = torch.deg2rad(y_pred)

        # Calculate the circular difference
        diff = torch.atan2(torch.sin(y_true_rad - y_pred_rad), torch.cos(y_true_rad - y_pred_rad))

        # Convert back to degrees and take the absolute value
        diff_deg = torch.rad2deg(diff)
        abs_diff_deg = torch.abs(diff_deg)

        return torch.sum(abs_diff_deg)

class Accuracy(torch.nn.Module):
    def __init__(self):
        super(Accuracy, self).__init__()

    def forward(self, y_true, y_pred):
        return torch.sum(y_true == y_pred)

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)