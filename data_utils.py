from os import listdir
from os.path import join

from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize, Lambda

import numpy as np
import cv2


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


def train_hr_transform(crop_size):
    return Compose([
        Resize(crop_size * 2),
        RandomCrop(crop_size),
        ToTensor(),
    ])

def val_hr_transform(crop_size):
    return Compose([
        Resize(crop_size),
        CenterCrop(crop_size),
        ToTensor(),
    ])

def noise(imagePill):
    image = np.array(imagePill.convert('YCbCr'))
    gaussian_noise = np.zeros((image.shape[0], image.shape[1]),dtype=np.uint8)
    cv2.randn(gaussian_noise, 128, 20)
    gaussian_noise = (gaussian_noise*0.5).astype(np.float32) - 64
    noisy_image = cv2.add(image[:,:,0].astype(np.float32),gaussian_noise)
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    image[:,:,0] = noisy_image
    imagePill = Image.fromarray(image, mode='YCbCr')
    return imagePill.convert('RGB')

def train_lr_transform(crop_size):
    return Compose([
        ToPILImage(),
        Lambda(noise),
        ToTensor()
    ])


def display_transform():
    return Compose([
        ToPILImage(),
        #Resize(400),
        CenterCrop(400),
        ToTensor()
    ])


class TrainDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, crop_size):
        super(TrainDatasetFromFolder, self).__init__()
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]
        self.hr_transform = train_hr_transform(crop_size)
        self.lr_transform = train_lr_transform(crop_size)

    def __getitem__(self, index):
        hr_image = self.hr_transform(Image.open(self.image_filenames[index]))
        lr_image = self.lr_transform(hr_image)
        return lr_image, hr_image

    def __len__(self):
        return len(self.image_filenames)



class ValDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, crop_size):
        super(ValDatasetFromFolder, self).__init__()
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]
        self.hr_transform = val_hr_transform(crop_size)
        self.lr_transform = train_lr_transform(crop_size)

    def __getitem__(self, index):
        hr_image = self.hr_transform(Image.open(self.image_filenames[index]))
        lr_image = self.lr_transform(hr_image)
        return lr_image, hr_image

    def __len__(self):
        return len(self.image_filenames)


class TestDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(TestDatasetFromFolder, self).__init__()
        self.lr_path = dataset_dir + '/SRF_' + str(upscale_factor) + '/data/'
        self.hr_path = dataset_dir + '/SRF_' + str(upscale_factor) + '/target/'
        self.upscale_factor = upscale_factor
        self.lr_filenames = [join(self.lr_path, x) for x in listdir(self.lr_path) if is_image_file(x)]
        self.hr_filenames = [join(self.hr_path, x) for x in listdir(self.hr_path) if is_image_file(x)]

    def __getitem__(self, index):
        image_name = self.lr_filenames[index].split('/')[-1]
        lr_image = Image.open(self.lr_filenames[index])
        w, h = lr_image.size
        hr_image = Image.open(self.hr_filenames[index])
        hr_scale = Resize((self.upscale_factor * h, self.upscale_factor * w), interpolation=Image.BICUBIC)
        hr_restore_img = hr_scale(lr_image)
        return image_name, ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image)

    def __len__(self):
        return len(self.lr_filenames)
