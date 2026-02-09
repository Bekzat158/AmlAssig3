import os
import torch
import numpy as np
import cv2
import requests
import zipfile
import shutil
from sklearn.model_selection import train_test_split

DATA_URL = 'https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip'
DATA_DIR = 'data'
ZIP_FILE = os.path.join(DATA_DIR, 'PennFudanPed.zip')
EXTRACTED_DIR = os.path.join(DATA_DIR, 'PennFudanPed')

def download_and_extract():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    if not os.path.exists(ZIP_FILE) and not os.path.exists(EXTRACTED_DIR):
        print("Downloading dataset...")
        r = requests.get(DATA_URL)
        with open(ZIP_FILE, 'wb') as f:
            f.write(r.content)
        print("Unzipping...")
        with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)
        print("Dataset ready.")
    else:
        print("Dataset extracted.")

def convert_to_yolo():
    # Create YOLO directory structure
    yolo_dir = os.path.join(DATA_DIR, 'yolo_data')
    if os.path.exists(yolo_dir):
        shutil.rmtree(yolo_dir)
    
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(yolo_dir, 'images', split))
        os.makedirs(os.path.join(yolo_dir, 'labels', split))

    imgs = list(sorted(os.listdir(os.path.join(EXTRACTED_DIR, "PNGImages"))))
    masks = list(sorted(os.listdir(os.path.join(EXTRACTED_DIR, "PedMasks"))))

    # Split: 70% train, 20% val, 10% test
    train_imgs, test_imgs, train_masks, test_masks = train_test_split(imgs, masks, test_size=0.3, random_state=42)
    val_imgs, test_imgs, val_masks, test_masks = train_test_split(test_imgs, test_masks, test_size=1/3, random_state=42) # 10% of total is 1/3 of 30%

    print(f"Split sizes: Train={len(train_imgs)}, Val={len(val_imgs)}, Test={len(test_imgs)}")

    def process_split(img_list, mask_list, split_name):
        for img_name, mask_name in zip(img_list, mask_list):
            img_path = os.path.join(EXTRACTED_DIR, "PNGImages", img_name)
            mask_path = os.path.join(EXTRACTED_DIR, "PedMasks", mask_name)
            
            # Copy image
            shutil.copy(img_path, os.path.join(yolo_dir, 'images', split_name, img_name))

            # Process mask to YOLO labels
            # YOLO format: class x_center y_center width height (normalized)
            mask = cv2.imread(mask_path, 0)
            obj_ids = np.unique(mask)
            obj_ids = obj_ids[1:] # remove background

            height, width = mask.shape
            
            label_path = os.path.join(yolo_dir, 'labels', split_name, os.path.splitext(img_name)[0] + '.txt')
            
            with open(label_path, 'w') as f:
                for obj_id in obj_ids:
                    pos = np.where(mask == obj_id)
                    xmin = np.min(pos[1])
                    xmax = np.max(pos[1])
                    ymin = np.min(pos[0])
                    ymax = np.max(pos[0])

                    # Calculate center and width/height
                    box_width = xmax - xmin
                    box_height = ymax - ymin
                    x_center = xmin + box_width / 2
                    y_center = ymin + box_height / 2

                    # Normalize
                    x_center /= width
                    y_center /= height
                    box_width /= width
                    box_height /= height
                    
                    # Class 0 for pedestrian
                    f.write(f"0 {x_center} {y_center} {box_width} {box_height}\n")

    process_split(train_imgs, train_masks, 'train')
    process_split(val_imgs, val_masks, 'val')
    process_split(test_imgs, test_masks, 'test')
    
    # Create dataset.yaml for YOLO
    yaml_content = f"""
path: {os.path.abspath(yolo_dir)}
train: images/train
val: images/val
test: images/test

names:
  0: pedestrian
"""
    with open(os.path.join(yolo_dir, 'dataset.yaml'), 'w') as f:
        f.write(yaml_content)

    print("YOLO data preparation complete.")

if __name__ == "__main__":
    download_and_extract()
    convert_to_yolo()
