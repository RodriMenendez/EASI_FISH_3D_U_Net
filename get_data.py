import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import transformations
import datasets
import util

def CreateDataset(ratios, image_shape, data_root, batch_size, num_workers, seed):
    train_idx, test_idx, val_idx = util.DataSplit(ratios, data_root, seed)

    image_resize = transformations.Resize3D(image_shape)
    label_resize = transforms.Compose([transformations.Resize3D(image_shape, "nearest"), transformations.BinaryReplace(0.0)])

    train_data = datasets.CustomDataset(data_root, train_idx, transform=image_resize, target_transform=label_resize)
    val_data = datasets.CustomDataset(data_root, val_idx, transform=image_resize, target_transform=label_resize)
    test_data = datasets.CustomDataset(data_root, test_idx, transform=image_resize, target_transform=label_resize)

    trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valloader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    testloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_data, trainloader, test_data, testloader, val_data, valloader
