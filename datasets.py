import torch
from torch.utils.data import Dataset
import numpy as np
import glob
from tifffile import imread
    
class CustomDataset(Dataset):
    def __init__(self, data_root, im_idxs, transform=None, target_transform=None):
        self.im_idxs = im_idxs
        self.image_paths = np.array(sorted(glob.glob(data_root + "images/*.tif")))[self.im_idxs]
        self.mask_paths = np.array(sorted(glob.glob(data_root + "labels/*.tif")))[self.im_idxs]
        self.exists = self.image_paths.size != 0 and self.mask_paths.size != 0
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.im_idxs)
    
    def __getitem__(self, idx):
        im = torch.tensor(imread(self.image_paths[idx]).astype(float)).unsqueeze(0)
        im_mask = torch.tensor(imread(self.mask_paths[idx]).astype(float)).unsqueeze(0)
        
        if self.transform:
            im = self.transform(im)

        if self.target_transform:
            im_mask = self.target_transform(im_mask)

        return im, im_mask
    
class CustomTensorDataset(Dataset):
    def __init__(self, data_root, im_idxs, transform=None, target_transform=None, spatial_dims=3):
        self.im_idxs = im_idxs
        image_path = glob.glob(data_root+"*inputs")
        mask_path = glob.glob(data_root+"*masks")
        self.images = torch.load(image_path[0])
        self.masks = torch.load(mask_path[0])
        if spatial_dims == 2:
            self.images = torch.reshape(self.images, (self.images.shape[0]*self.images.shape[1], self.images.shape[2], self.images.shape[3]))
            self.masks = torch.reshape(self.masks, (self.masks.shape[0]*self.masks.shape[1], self.masks.shape[2], self.masks.shape[3]))
        self.images = self.images[self.im_idxs]
        self.masks = self.masks[self.im_idxs]
        self.exists = self.images.shape[0] != 0 and self.masks.shape[0] != 0
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        im = self.images[idx].unsqueeze(0)
        im_mask = self.masks[idx].unsqueeze(0)
        
        if self.transform:
            im = self.transform(im)

        if self.target_transform:
            im_mask = self.target_transform(im_mask)

        return im, im_mask