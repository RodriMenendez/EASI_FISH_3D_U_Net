import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import transformations
import datasets
import util

def CreateDataset(ratios, image_shape, data_root, data_type, data_len, batch_size, num_workers, seed, spatial_dims=3):
    train_idx, test_idx, val_idx = util.DataSplit(ratios, data_root, seed, data_len)

    if spatial_dims == 3:
        image_resize = transformations.Resize3D(image_shape)
        label_resize = transforms.Compose([transformations.Resize3D(image_shape, "nearest"), transformations.BinaryReplace(0.0)])
    elif spatial_dims == 2:
        image_resize = transforms.Resize(image_shape, antialias=True)
        label_resize = transforms.Compose([transforms.Resize(image_shape, transforms.InterpolationMode.NEAREST_EXACT), transformations.BinaryReplace(0.0)])

    if data_type == 'tif':
        train_data = datasets.CustomDataset(data_root, train_idx, transform=image_resize, target_transform=label_resize)
        val_data = datasets.CustomDataset(data_root, val_idx, transform=image_resize, target_transform=label_resize)
        test_data = datasets.CustomDataset(data_root, test_idx, transform=image_resize, target_transform=label_resize)
    elif data_type == 'tensor':
        train_data = datasets.CustomTensorDataset(data_root, train_idx, transform=image_resize, target_transform=label_resize, spatial_dims=spatial_dims)
        val_data = datasets.CustomTensorDataset(data_root, val_idx, transform=image_resize, target_transform=label_resize, spatial_dims=spatial_dims)
        test_data = datasets.CustomTensorDataset(data_root, test_idx, transform=image_resize, target_transform=label_resize, spatial_dims=spatial_dims)

    trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valloader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    testloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_data, trainloader, test_data, testloader, val_data, valloader
