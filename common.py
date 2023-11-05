import argparse
import numpy as np
import torch
import os

def define_args():
    parser = argparse.ArgumentParser()

    # Learning Hyperparameters
    parser.add_argument('--epochs', type = int, default = 100)
    parser.add_argument('--lr', type = float, default = 5e-5)
    parser.add_argument('--batch_size', type = int, default = 512)
    parser.add_argument('--num_workers', type = int, default = 4)
    parser.add_argument('--scheduler', choices=['step', 'onecycle'], default='step')

    # Model Hyperparameters
    parser.add_argument('--model', type = str, default = 'convnext-multi-norm')
    parser.add_argument('--pretrained', type = bool, default = False)
    
    # Trainer Hyperparameters
    parser.add_argument('--trainer', choices=['regression', 'classification', 'multi'], default='multi')
    parser.add_argument('--augmentation', choices=['no-aug', 'weak-aug', 'rand-aug', 'auto-aug'], default='weak-aug')
    parser.add_argument('--mixup', action='store_true', default=False)
    parser.add_argument('--cutmix', action='store_true', default=False)
    parser.add_argument('--dataset', choices=['angle'], default='angle')
    parser.add_argument('--magnitude', type=int, default=5)

    # Misc Hyperparameters
    parser.add_argument('--wandb_log', type=bool, default=False)
    parser.add_argument('--exp_index', type=int, default=1)
    parser.add_argument('--checkpoint', type=str, default = '')
    parser.add_argument('--run_name', type=str, default = '')
    parser.add_argument('--dataset_path', type=str, default = '../angle_dataset/')
    parser.add_argument('--seed', type=int, default=21)
    args = parser.parse_args()

    return args

def load_checkpoint(model, weight_path):
    if os.path.exists(weight_path):
        print('Loading checkpoint from {}'.format(weight_path))
        model.load_state_dict(torch.load(weight_path))

def save_checkpoint(model, weight_path):
    print('Saving checkpoint to {}'.format(weight_path))
    torch.save(model.state_dict(), weight_path)