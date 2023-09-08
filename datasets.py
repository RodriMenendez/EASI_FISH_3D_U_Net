import torch
from torch.utils.data import Dataset
import numpy as np
import glob
from tifffile import imread
    
class CustomTifDataset(Dataset):
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
    
class CustomVaa3DDataset(Dataset):
    def __init__(self, masks_path, inputs_path, idxs, transform=None, target_transform=None, spatial_dims=3, depth=32):
        if spatial_dims == 2:
            self.idxs = np.unique(idxs//depth).astype(int)
        else:
            self.idxs = idxs
        self.inputs_paths = np.array(sorted(glob.glob(inputs_path+"**/*[!output].tif", recursive=True)))[self.idxs]
        self.inputs_paths, self.masks_paths = self.get_existing_data_paths(masks_path, self.inputs_paths)
        self.transform = transform
        self.target_transform = target_transform
        self.exists = len(self.inputs_paths) != 0 and len(self.masks_paths) != 0
        self.depth = depth
        self.spatial_dims = spatial_dims

    def get_mask_path(self, masks_path, input_path):
        name_match = input_path.split('/')[-1].split('.')[0]
        mask_file_name = glob.glob(f'{masks_path}/*{name_match}*')

        return mask_file_name

    def get_existing_data_paths(self, masks_path, input_paths):
        masks_paths_exists = []
        inputs_paths_exists = []

        for input_path in input_paths:
            mask_path_file = self.get_mask_path(masks_path, input_path)
            if len(mask_path_file) != 0:
                masks_paths_exists.append(mask_path_file)
                inputs_paths_exists.append(input_path)

        return inputs_paths_exists, masks_paths_exists

    def __len__(self):
        if self.spatial_dims == 2:
            length = self.depth*len(self.masks_paths)
        else:
            length = len(self.masks_paths)

        return length

    def __getitem__(self, idx):
        idx_3d = idx//self.depth
        idx_2d = idx%self.depth
        im = torch.tensor(imread(self.inputs_paths[idx_3d]).astype(float))
        im_mask = torch.tensor(imread(self.masks_paths[idx_3d]).astype(float))

        if self.spatial_dims == 3:
            im = im.unsqueeze(0)
            im_mask = im_mask.unsqueeze(0)
        else:
            im = im[idx_2d].unsqueeze(0).to(float)
            im_mask = im_mask[idx_2d].unsqueeze(0)
        
        if self.transform:
            im = self.transform(im)

        if self.target_transform:
            im_mask = self.target_transform(im_mask)

        return im, im_mask