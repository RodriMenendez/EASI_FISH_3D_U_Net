import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import transformations
from datasets import *
import util

def CreateDataset(ratios, image_shape, data_root, masks_path, inputs_path, custom_dataset, transforms, data_len, batch_size, num_workers, seed, spatial_dims=3):
    train_idx, test_idx, val_idx = util.DataSplit(ratios, data_root, seed, data_len, spatial_dims, image_shape[0])

    if spatial_dims == 3:
        if transforms:
            image_resize = transformations.CustomTransformation(transforms, image_shape)
            label_resize = transformations.CustomTransformation(transforms, image_shape, True)
        else:
            image_resize = transformations.Resize3D(image_shape)
            label_resize = transforms.Compose([transformations.Resize3D(image_shape, "nearest"), transformations.BinaryReplace(0.0)])
    elif spatial_dims == 2:
        image_resize = transforms.Resize(image_shape[1:], antialias=True)
        label_resize = transforms.Compose([transforms.Resize(image_shape[1:], transforms.InterpolationMode.NEAREST_EXACT), transformations.BinaryReplace(0.0)])

    if custom_dataset == 'tif':
        train_data = CustomTifDataset(data_root, train_idx, transform=image_resize, target_transform=label_resize)
        val_data = CustomTifDataset(data_root, val_idx, transform=image_resize, target_transform=label_resize)
        test_data = CustomTifDataset(data_root, test_idx, transform=image_resize, target_transform=label_resize)
    elif custom_dataset == 'tensor':
        train_data = CustomTensorDataset(data_root, train_idx, transform=image_resize, target_transform=label_resize, spatial_dims=spatial_dims)
        val_data = CustomTensorDataset(data_root, val_idx, transform=image_resize, target_transform=label_resize, spatial_dims=spatial_dims)
        test_data = CustomTensorDataset(data_root, test_idx, transform=image_resize, target_transform=label_resize, spatial_dims=spatial_dims) 
    elif custom_dataset == 'vaa3d':
        depth = image_shape[0]
        train_data = CustomVaa3DDataset(masks_path, inputs_path, train_idx, transform=image_resize, target_transform=label_resize, spatial_dims=spatial_dims, depth=depth)
        val_data = CustomVaa3DDataset(masks_path, inputs_path, val_idx, transform=image_resize, target_transform=label_resize, spatial_dims=spatial_dims, depth=depth)
        test_data = CustomVaa3DDataset(masks_path, inputs_path, test_idx, transform=image_resize, target_transform=label_resize, spatial_dims=spatial_dims, depth=depth) 
    else:
        data = torch.load(custom_dataset)
        train_data, test_data, val_data = data

    trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valloader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    testloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_data, trainloader, test_data, testloader, val_data, valloader
