import os
import cv2
import shutil
import argparse

def define_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_path', type=str, default='../CFV-Dataset/')
    args = parser.parse_args()

    return args

args = define_args()

output_path = '../angle_dataset/'

train_csv_path = os.path.join(args.dataset_path, 'train.csv')
test_csv_path = os.path.join(args.dataset_path, 'test.csv')

output_train_csv_path = os.path.join(output_path, 'train.csv')
output_test_csv_path = os.path.join(output_path, 'test.csv')

os.makedirs(output_path, exist_ok=True)

shutil.copy(train_csv_path, output_train_csv_path)
shutil.copy(test_csv_path, output_test_csv_path)

input_image_dir = os.path.join(args.dataset_path, 'images/')
output_image_dir = os.path.join(output_path, 'images/')
for id_dir in sorted(os.listdir(input_image_dir)):
    input_id_dir = os.path.join(input_image_dir, id_dir)
    output_id_dir = os.path.join(output_image_dir, id_dir)
    os.makedirs(output_id_dir, exist_ok=True)

    for image_name in sorted(os.listdir(input_id_dir)):
        input_image_path = os.path.join(input_id_dir, image_name)
        output_image_path = os.path.join(output_id_dir, image_name)

        image = cv2.imread(input_image_path)
        image = cv2.resize(image, (224, 224))

        cv2.imwrite(output_image_path, image)
