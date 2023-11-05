# Official Repository for "CFV Dataset: Fine-Grained Predictions of Car Orientation from Images"

## Prerequisites
Install all dependencies with the following command:
```pip install -r requirements.txt```

## Usage

1. Download data from [here](https://drive.google.com/drive/folders/1tQh9p4P9Xt_40eJCCwfz2zyESScVIyd7?usp=drive_link) and extract it.
2. Run data preparation script: ```python scripts/prepare_data.py --dataset_path /path/to/CFV_Dataset ```
3. To run all the experiments from the paper: ```scripts/run_all_experiments.sh```
4. To train the top-performing model: ```scripts/train_top_mode.sh```

The weights of the pretrained top-performing model can be found [here](https://drive.google.com/drive/folders/1tQh9p4P9Xt_40eJCCwfz2zyESScVIyd7?usp=drive_link)

## Citation

TBA

## Licence & Acknowledgement
The CFV Dataset is released under the Apache 2.0 License (see LICENSE).
