from __future__ import print_function, division
import os
from typing import Optional

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import glob
import os
import cv2
import json
from util.preprocess import preprocess_unityeyes_image


class UnityEyesDataset(Dataset):
    def __init__(self, img_dir: str, json_dir: str):
        """
        Args:
            img_dir (str): Path to the 'TestSet' directory which contains
                           subfolders like BottomCenter, TopLeft, etc.
            json_dir (str): Path to the 'TestSet_json' directory which contains
                            all the JSON files (e.g., 18.json).
        """
        self.img_dir = img_dir
        self.json_dir = json_dir

        # Search all subdirectories under img_dir for .jpg images
        self.img_paths = glob.glob(os.path.join(self.img_dir, "*", "*.jpg"))
        # Sort by numeric index extracted from filename
        # Assumes filenames are something like '18.jpg'
        self.img_paths = sorted(
            self.img_paths, key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
        )

        # Create corresponding json paths
        self.json_paths = []
        for img_path in self.img_paths:
            idx = os.path.splitext(os.path.basename(img_path))[0]
            json_path = os.path.join(self.json_dir, f"{idx}.json")
            if not os.path.exists(json_path):
                raise FileNotFoundError(
                    f"JSON file not found for image {img_path}: {json_path}"
                )
            self.json_paths.append(json_path)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.img_paths[idx]
        json_path = self.json_paths[idx]

        full_img = cv2.imread(img_path)
        if full_img is None:
            raise ValueError(f"Could not read image: {img_path}")

        with open(json_path, "r") as f:
            json_data = json.load(f)

        # Preprocess image and JSON data
        eye_sample = preprocess_unityeyes_image(full_img, json_data)
        sample = {"full_img": full_img, "json_data": json_data}
        sample.update(eye_sample)
        return sample
