#!/bin/bash

# Class
python main.py --wandb_log True --pretrained True --run_name resnet-class --epochs 100 --exp_index 1 --batch_size 512 --lr 5e-5 --augmentation rand-aug --magnitude 5 --model resnet18-class --trainer classification
python main.py --wandb_log True --pretrained True --run_name convnext-class --epochs 100 --exp_index 1 --batch_size 512 --lr 5e-5 --augmentation rand-aug --magnitude 5 --model convnext-class --trainer classification
python main.py --wandb_log True --pretrained True --run_name swin-class --epochs 100 --exp_index 1 --batch_size 512 --lr 5e-5 --augmentation rand-aug --magnitude 5 --model swin-class --trainer classification

python main.py --wandb_log True --run_name init-resnet-class --epochs 100 --exp_index 1 --batch_size 512 --lr 5e-5 --augmentation rand-aug --magnitude 5 --model resnet18-class --trainer classification
python main.py --wandb_log True --run_name init-convnext-class --epochs 100 --exp_index 1 --batch_size 512 --lr 5e-5 --augmentation rand-aug --magnitude 5 --model convnext-class --trainer classification
python main.py --wandb_log True --run_name init-swin-class --epochs 100 --exp_index 1 --batch_size 512 --lr 5e-5 --augmentation rand-aug --magnitude 5 --model swin-class --trainer classification

# Regression Rand
python main.py --wandb_log True --pretrained True --run_name resnet-rand --epochs 100 --exp_index 1 --batch_size 512 --lr 5e-5 --augmentation rand-aug --magnitude 5 --model resnet18-norm
python main.py --wandb_log True --pretrained True --run_name convnext-rand --epochs 100 --exp_index 1 --batch_size 512 --lr 5e-5 --augmentation rand-aug --magnitude 5 --model convnext-small-norm
python main.py --wandb_log True --pretrained True --run_name swin-rand --epochs 100 --exp_index 1 --batch_size 512 --lr 5e-5 --augmentation rand-aug --magnitude 5 --model swin-small-norm

python main.py --wandb_log True --run_name init-resnet-rand --epochs 100 --exp_index 1 --batch_size 512 --lr 5e-5 --augmentation rand-aug --magnitude 5 --model resnet18-norm
python main.py --wandb_log True --run_name init-convnext-rand --epochs 100 --exp_index 1 --batch_size 512 --lr 5e-5 --augmentation rand-aug --magnitude 5 --model convnext-small-norm
python main.py --wandb_log True --run_name init-swin-rand --epochs 100 --exp_index 1 --batch_size 512 --lr 5e-5 --augmentation rand-aug --magnitude 5 --model swin-small-norm

# Multi Weak
python main.py --wandb_log True --pretrained True --run_name resnet-multi-weak --epochs 100 --exp_index 1 --batch_size 512 --lr 5e-5 --augmentation weak-aug --magnitude 5 --model resnet18-multi-norm --trainer multi
python main.py --wandb_log True --pretrained True --run_name convnext-multi-weak --epochs 100 --exp_index 1 --batch_size 512 --lr 5e-5 --augmentation weak-aug --magnitude 5 --model convnext-multi-norm --trainer multi
python main.py --wandb_log True --pretrained True --run_name swin-multi-weak --epochs 100 --exp_index 1 --batch_size 512 --lr 5e-5 --augmentation weak-aug --magnitude 5 --model swin-multi-norm --trainer multi

python main.py --wandb_log True --run_name init-resnet-multi-weak --epochs 100 --exp_index 1 --batch_size 512 --lr 5e-5 --augmentation weak-aug --magnitude 5 --model resnet18-multi-norm --trainer multi
python main.py --wandb_log True --run_name init-convnext-multi-weak --epochs 100 --exp_index 1 --batch_size 512 --lr 5e-5 --augmentation weak-aug --magnitude 5 --model convnext-multi-norm --trainer multi
python main.py --wandb_log True --run_name init-swin-multi-weak --epochs 100 --exp_index 1 --batch_size 512 --lr 5e-5 --augmentation weak-aug --magnitude 5 --model swin-multi-norm --trainer multi

# Regression Weak
python main.py --wandb_log True --pretrained True --run_name resnet-weak --epochs 100 --exp_index 1 --batch_size 512 --lr 5e-5 --augmentation weak-aug --magnitude 5 --model resnet18-norm
python main.py --wandb_log True --pretrained True --run_name convnext-weak --epochs 100 --exp_index 1 --batch_size 512 --lr 5e-5 --augmentation weak-aug --magnitude 5 --model convnext-small-norm
python main.py --wandb_log True --pretrained True --run_name swin-weak --epochs 100 --exp_index 1 --batch_size 512 --lr 5e-5 --augmentation weak-aug --magnitude 5 --model swin-small-norm

python main.py --wandb_log True --run_name init-resnet-weak --epochs 100 --exp_index 1 --batch_size 512 --lr 5e-5 --augmentation weak-aug --magnitude 5 --model resnet18-norm
python main.py --wandb_log True --run_name init-convnext-weak --epochs 100 --exp_index 1 --batch_size 512 --lr 5e-5 --augmentation weak-aug --magnitude 5 --model convnext-small-norm
python main.py --wandb_log True --run_name init-swin-weak --epochs 100 --exp_index 1 --batch_size 512 --lr 5e-5 --augmentation weak-aug --magnitude 5 --model swin-small-norm

# Regression No Aug
python main.py --wandb_log True --pretrained True --run_name resnet-no-aug --epochs 100 --exp_index 1 --batch_size 512 --lr 5e-5 --augmentation no-aug --magnitude 5 --model resnet18-norm
python main.py --wandb_log True --pretrained True --run_name convnext-no-aug --epochs 100 --exp_index 1 --batch_size 512 --lr 5e-5 --augmentation no-aug --magnitude 5 --model convnext-small-norm
python main.py --wandb_log True --pretrained True --run_name swin-no-aug --epochs 100 --exp_index 1 --batch_size 512 --lr 5e-5 --augmentation no-aug --magnitude 5 --model swin-small-norm

python main.py --wandb_log True --run_name init-resnet-no-aug --epochs 100 --exp_index 1 --batch_size 512 --lr 5e-5 --augmentation no-aug --magnitude 5 --model resnet18-norm
python main.py --wandb_log True --run_name init-convnext-no-aug --epochs 100 --exp_index 1 --batch_size 512 --lr 5e-5 --augmentation no-aug --magnitude 5 --model convnext-small-norm
python main.py --wandb_log True --run_name init-swin-no-aug --epochs 100 --exp_index 1 --batch_size 512 --lr 5e-5 --augmentation no-aug --magnitude 5 --model swin-small-norm