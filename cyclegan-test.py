import argparse
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader
from models import *
from datasets_test import *
from utils import *
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=100, help="epoch to start training from")
parser.add_argument("--dataset_name", type=str, default="vi2ir_cs", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=480, help="size of image height")
parser.add_argument("--img_width", type=int, default=640, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--n_residual_blocks", type=int, default=6, help="number of residual blocks in generator")
parser.add_argument("--lambda_cyc", type=float, default=10.0, help="cycle loss weight")
parser.add_argument("--lambda_id", type=float, default=5.0, help="identity loss weight")
opt = parser.parse_args()

print(opt)

# Create sample and checkpoint directories
os.makedirs("images/%s" % opt.dataset_name, exist_ok=True)
os.makedirs("saved_models/%s" % opt.dataset_name, exist_ok=True)

cuda = torch.cuda.is_available()

input_shape = (opt.channels, opt.img_height, opt.img_width)

# Initialize generator and discriminator
G_AB = GeneratorResNet(input_shape, opt.n_residual_blocks)

if cuda:
    G_AB = G_AB.cuda()

if opt.epoch != 0:
    # Load pretrained models
    G_AB.load_state_dict(torch.load("saved_models/%s/G_AB_%d.pth" % (opt.dataset_name, opt.epoch)))

else:
    # Initialize weights
    G_AB.apply(weights_init_normal)

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

# Image transformations
transforms_ = [
    transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
    # transforms.RandomCrop((opt.img_height, opt.img_width)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

# Test data loader
val_dataloader = DataLoader(
    ImageDataset("../../data/%s" % opt.dataset_name, transforms_=transforms_, unaligned=True, mode="test"),
    batch_size=2,
    shuffle=False,
    num_workers=0,
)


def sample_images(batches_done):
    """Saves a generated sample from the test set"""
    imgs = next(iter(val_dataloader))
    G_AB.eval()
    real_RGB = Variable(imgs["RGB"].type(Tensor))
    fake_RGB = G_AB(real_RGB)
    real_Gray = Variable(imgs["Gray"].type(Tensor))
    fake_Gray = G_AB(real_Gray)
    IR = Variable(imgs["IR"].type(Tensor))
    # Arange images along x-axis
    real_RGB = make_grid(real_RGB, nrow=4, normalize=True)
    fake_RGB = make_grid(fake_RGB, nrow=4, normalize=True)
    real_Gray = make_grid(real_Gray, nrow=4, normalize=True)
    fake_Gray = make_grid(fake_Gray, nrow=4, normalize=True)
    IR = make_grid(IR, nrow=4, normalize=True)
    # Arange images along y-axis
    image_grid = torch.cat((real_RGB, fake_RGB, real_Gray, fake_Gray, IR), 1)
    save_image(image_grid, "images/%s/test_%s.png" % ('test_results', batches_done), normalize=False)


# ----------
#  Training
# ----------
if __name__ == '__main__':
    start_time = time.time()

    batches_done = opt.epoch
    # If at sample interval save image
    sample_images(batches_done)

    end_time = time.time()
    total_time = end_time - start_time
    print(f"程序的运行时间为 {total_time:.3f} 秒")