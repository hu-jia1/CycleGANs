import glob
import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode="train"):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        self.files_RGB = sorted(glob.glob(os.path.join(root, mode) + "/rgb+IR+Gray/*rgb.*"))
        self.files_Gray = sorted(glob.glob(os.path.join(root, mode) + "/rgb+IR+Gray/*gray.*"))
        self.files_IR = sorted(glob.glob(os.path.join(root, mode) + "/rgb+IR+Gray/*IR.*"))

    def __getitem__(self, index):

        img_RGB = Image.open(self.files_RGB[index % len(self.files_RGB)])
        img_Gray = Image.open(self.files_Gray[index % len(self.files_RGB)])
        img_IR = Image.open(self.files_IR[index % len(self.files_RGB)])

        # 判断图像是否3通道
        if img_Gray.mode != 'RGB':
            # 将单通道图像转换为3通道图像
            img_3_Gray = img_Gray.convert('RGB')
        else:
            img_3_Gray = img_Gray

        if img_IR.mode != 'RGB':
            img_3_IR = img_IR.convert('RGB')
        else:
            img_3_IR = img_IR

        img_RGB = self.transform(img_RGB)
        img_3_Gray = self.transform(img_3_Gray)
        img_3_IR = self.transform(img_3_IR)

        return {"RGB": img_RGB, "Gray": img_3_Gray, "IR":img_3_IR}

    def __len__(self):
        return len(self.files_RGB)