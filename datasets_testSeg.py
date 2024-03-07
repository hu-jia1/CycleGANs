import glob
import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode="train"):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        self.files_RGB = sorted(glob.glob(os.path.join(root, mode) + "/*.*"))
        # self.files_IR = sorted(glob.glob(os.path.join(root, mode) + "/*.*"))

    def __getitem__(self, index):

        file = self.files_RGB[index % len(self.files_RGB)]
        filename = file.split('\\')[-1]

        img_RGB = Image.open(file)
        # img_IR = Image.open(self.files_IR[index % len(self.files_RGB)])


        if img_RGB.mode != 'RGB':
            img_3_IR = img_RGB.convert('RGB')
        else:
            img_3_IR = img_RGB

        img_RGB = self.transform(img_3_IR)


        # img_3_IR = self.transform(img_3_IR)

        return {"RGB": img_RGB,'filename':filename}

    def __len__(self):
        return len(self.files_RGB)