#!/usr/bin/env python3

import glob
import random
from skimage import io
import torchvision
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF



class CustomRotationTransform:
    """Rotate by one of the given angles."""

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)

rotation_transform = CustomRotationTransform(angles=[0, 90, 180,270])

### Whole dataset is cached and available in self.cached_data
class GalaxyDataset(Dataset):
    
    def __init__(self, filespath,prefix,use_cache = False, transform = None):
        
        self.prefix      = prefix        
        self.labels      = {}
        self.filenames   = []
        self.transform   = transform
        self.cached_data = []
        self.use_cache   = use_cache
        self.train_transforms = []


        if prefix == "train":
            train_transforms = []
            train_transforms.append(torchvision.transforms.ColorJitter(brightness=(0,1.2), contrast = (0.2, 1.8)))
            train_transforms.append(torchvision.transforms.RandomHorizontalFlip(p=0.5))
            train_transforms.append(torchvision.transforms.RandomVerticalFlip(p=0.5))
#            train_transforms.append(rotation_transform)
            self.train_transforms = torchvision.transforms.Compose(train_transforms)
        
        if prefix == "trainval":
            train_transforms = []
            train_transforms.append(torchvision.transforms.ColorJitter(brightness=(0,1.2), contrast = (0.2, 1.8)))
            train_transforms.append(torchvision.transforms.RandomHorizontalFlip(p=0.5))
            train_transforms.append(torchvision.transforms.RandomVerticalFlip(p=0.5))
#            train_transforms.append(rotation_transform)
            self.train_transforms = torchvision.transforms.Compose(train_transforms)
     
        class_0_files = []
        class_1_files = []
        class_2_files = []
        class_3_files = []
        class_4_files = []

        if prefix == "trainval":
            class_0_files = glob.glob(filespath + "train" + "_class_0_*.jpg") + glob.glob(filespath + "val" + "_class_0_*.jpg")
            class_1_files = glob.glob(filespath + "train" + "_class_1_*.jpg") + glob.glob(filespath + "val" + "_class_1_*.jpg")
            class_2_files = glob.glob(filespath + "train" + "_class_2_*.jpg") + glob.glob(filespath + "val" + "_class_2_*.jpg")
            class_3_files = glob.glob(filespath + "train" + "_class_3_*.jpg") + glob.glob(filespath + "val" + "_class_3_*.jpg")
            class_4_files = glob.glob(filespath + "train" + "_class_4_*.jpg") + glob.glob(filespath + "val" + "_class_4_*.jpg")
        else:
            class_0_files = glob.glob(filespath + prefix + "_class_0_*.jpg")
            class_1_files = glob.glob(filespath + prefix + "_class_1_*.jpg")
            class_2_files = glob.glob(filespath + prefix + "_class_2_*.jpg")
            class_3_files = glob.glob(filespath + prefix + "_class_3_*.jpg")
            class_4_files = glob.glob(filespath + prefix + "_class_4_*.jpg")           


        self.filenames = class_0_files + class_1_files + class_2_files + class_3_files + class_4_files

        self.__datasetup__(class_0_files,0)
        self.__datasetup__(class_1_files,1)
        self.__datasetup__(class_2_files,2)
        self.__datasetup__(class_3_files,3)
        self.__datasetup__(class_4_files,4)
        
        random.shuffle(self.filenames)
    
    def __datasetup__(self,files, label):
        for filename in files:
            self.labels[filename] = label
            self.filenames.append(filename)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        
        if not self.use_cache:
            filename = self.filenames[idx]
            label    = self.labels[filename]
            img      = io.imread(filename)
            sample   = {"image" : img,"label": label}
            self.cached_data.append(sample)        
        else:
            sample = self.cached_data[idx]    
        if self.transform:
            sample = self.transform(sample)
        if self.prefix == "train":
            sample["image"] = self.train_transforms(sample["image"])

        return sample

    def set_use_cache(self, use_cache):
        if use_cache:
            self.cached_data = self.cached_data
        else:
            self.cached_data = []
        self.use_cache = use_cache
