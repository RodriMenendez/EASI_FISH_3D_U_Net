import torch
from torch.utils.data import Dataset
import numpy as np
import glob

class EFDataset(Dataset):
    def __init__(self, data_root, im_idxs, transform=None, target_transform=None):
        self.im_idxs = im_idxs
        self.image_paths = np.array(sorted(glob.glob(data_root + "images/*.pt")))[self.im_idxs]
        self.mask_paths = np.array(sorted(glob.glob(data_root + "masks/*.pt")))[self.im_idxs]
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.im_idxs)

    def __getitem__(self, idx):
        im = torch.load(self.image_paths[idx]).float()
        im_mask = torch.load(self.mask_paths[idx]).float()

        if self.transform:
            im = self.transform(im)

        if self.target_transform:
            im_mask = self.target_transform(im_mask)


        return im, im_mask