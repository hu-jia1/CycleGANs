import glob
import random
import os
import numpy as np

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode="train"):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(root, mode) + "/Infrared/*.*"))
        self.files_B = sorted(glob.glob(os.path.join(root, mode) + "/RGB/*.*"))

    def __getitem__(self, index):
        a = self.files_A[index % len(self.files_A)]
        b = self.files_B[index % len(self.files_B)]
        img_A = Image.open(self.files_A[index % len(self.files_A)])
        img_B = Image.open(self.files_B[index % len(self.files_B)])

        # 判断图像是否3通道
        if img_A.mode != 'RGB':
            # 将单通道图像转换为3通道图像
            img_A_RGB = img_A.convert('RGB')
        else:
            img_A_RGB = img_A

        if img_B.mode != 'RGB':
            img_B_RGB = img_B.convert('RGB')
        else:
            img_B_RGB = img_B

        # 按0.5的概率对图像进行水平翻转
        if np.random.random() < 0.5:
            img_A_RGB = Image.fromarray(np.array(img_A_RGB)[:, ::-1, :], "RGB")
            img_B_RGB = Image.fromarray(np.array(img_B_RGB)[:, ::-1, :], "RGB")

        img_A_RGB = self.transform(img_A_RGB)
        img_B_RGB = self.transform(img_B_RGB)

        return {"A": img_A_RGB, "B": img_B_RGB}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))