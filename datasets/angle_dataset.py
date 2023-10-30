import os
import pandas as pd
from torch.utils.data.dataset import Dataset
from PIL import Image

class AngleDataset(Dataset):
    def __init__(self, args, csv_path, transform):
        self.transform = transform
        self.data = pd.read_csv(csv_path)
        self.data_path = os.path.join(args.dataset_path, 'images/')
    
    def __len__(self):
        return len(self.data)

    def get_angle_class(self, angle):
        # Front, Front left, Left, Back left, Back, Back right, Right, Front right
        if angle >= 337.5 or angle <= 22.5:
            return 0
        elif angle > 22.5 and angle <= 67.5:
            return 1
        elif angle > 67.5 and angle <= 112.5:
            return 2
        elif angle > 112.5 and angle <= 157.5:
            return 3
        elif angle > 157.5 and angle <= 202.5:
            return 4
        elif angle > 202.5 and angle <= 247.5:
            return 5
        elif angle > 247.5 and angle <= 292.5:
            return 6
        elif angle > 292.5 and angle <= 337.5:
            return 7

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        img_path = os.path.join(self.data_path, row.image_path)
        label = row.angle
        class_label = self.get_angle_class(label)
        img = Image.open(img_path)

        normalized_label = label / 360.0
        
        if self.transform:
            img = self.transform(img)
        
        output = {'img': img, 'label': label, 'normalized_label': normalized_label, 'class_label': class_label}

        return output