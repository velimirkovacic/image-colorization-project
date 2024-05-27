import os
from PIL import Image
from torchvision import transforms
import torch
import glob
from skimage import color


def convert_to_lab(img_path, transform):
    img = Image.open(img_path).convert("RGB")
    img = color.rgb2lab(img).astype("float32")
    img = transform(img)
    img[0,:,:] = img[0,:,:] / 50.0 - 1.0
    img[1:,:,:] = img[1:,:,:] / 110.0
    pre, ext = os.path.splitext(img_path)
    torch.save(img, pre + ".pt")



transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256), antialias=True)
])

path_train = "../data/train2017"
path_test = "../data/test2017"
path_now = path_test

paths = glob.glob(path_now + "/*.jpg")

task = [i for i in range(0, 10000)]
task = [i for i in range(10000, 20000)]
task = [i for i in range(20000, 30000)]
task = [i for i in range(30000, len(paths))]

for i, curr_img in enumerate(paths):
    if i in task:
        convert_to_lab(curr_img, transform)
        if (i + 1) % 100 == 0:
            print(f"Converted {i + 1} of {len(paths)} images.")