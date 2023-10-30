#!/bin/bash

python main.py --wandb_log True --pretrained True --run_name convnext-multi-weak --epochs 100 --exp_index 1 --batch_size 512 --lr 5e-5 --augmentation weak-aug --magnitude 5 --model convnext-multi-norm --trainer multi
