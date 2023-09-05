import glob

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import transformations
from datasets import CustomVaa3DDataset
import util

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_save_path', type=str, default='data', help='filepath of pickle file that data will be saved to')
parser.add_argument('--masks_path', type=str, required=True, help='filepath for folder where masks are located')
parser.add_argument('--inputs_path', type=str, required=True, help='filepath for folder where input images ae located')
parser.add_argument('--data_split_ratios', nargs=3, default= [0.8, 0.1, 0.1], type=float, help='ratio of train to test to validation data (defaults to [0.8, 0.1, 0.1])')
parser.add_argument('--image_shape', nargs='+', default = [64, 128, 128], type=int, help='z x y shape to resize images to (defaults to [64, 128, 128])')
parser.add_argument('--data_len', type=int, default=None, help='length of dataset (defaults to None)')
parser.add_argument('--batch_size', type=int, default=1, help='batch size for training (defaults to 1)')
parser.add_argument('--num_workers', default=1, type=int, help='number of workers for dataloader (defaults to 1)')
parser.add_argument('--spatial_dims', default=3, type=int, help='number of spatial dimensions of model (defaults to 3)')
parser.add_argument('--seed', type=int, default=0, help='seed for random data splitting (defaults to 0)')

args = parser.parse_args()

def CreateDataset(datasets_save_path, ratios, image_shape, masks_path, inputs_path, data_len, batch_size, num_workers, seed, spatial_dims=3):
    if not data_len:
        data_len = len(glob.glob(inputs_path+"**/*[!output].tif", recursive=True))

    train_idx, test_idx, val_idx = util.DataSplit(ratios, None, seed, data_len, spatial_dims, image_shape[0])

    if spatial_dims == 3:
        image_resize = transformations.Resize3D(image_shape)
        label_resize = transforms.Compose([transformations.Resize3D(image_shape, "nearest"), transformations.BinaryReplace(0.0)])
    elif spatial_dims == 2:
        image_resize = transforms.Resize(image_shape[1:], antialias=True)
        label_resize = transforms.Compose([transforms.Resize(image_shape[1:], transforms.InterpolationMode.NEAREST_EXACT), transformations.BinaryReplace(0.0)])

    train_data = CustomVaa3DDataset(masks_path, inputs_path, train_idx, transform=image_resize, target_transform=label_resize)
    val_data = CustomVaa3DDataset(masks_path, inputs_path, val_idx, transform=image_resize, target_transform=label_resize)
    test_data = CustomVaa3DDataset(masks_path, inputs_path, test_idx, transform=image_resize, target_transform=label_resize)

    data = (train_data, test_data, val_data)
    torch.save(data, datasets_save_path)

def main(args):
    CreateDataset(args.dataset_save_path,
                         args.data_split_ratios,
                         args.image_shape,
                         args.masks_path,
                         args.inputs_path,
                         args.data_len,
                         args.batch_size,
                         args.num_workers,
                         args.seed,
                         args.spatial_dims)

if __name__ == '__main__':
    main(args)


