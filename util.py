import numpy as np
import glob

def DataSplit(ratios, data_root, seed, data_len=None, spatial_dims=3, z=32):
    if not data_len:
        data_len = len(glob.glob(data_root + 'images/*.tif'))
    idx = np.arange(data_len)
    np.random.seed(seed)
    np.random.shuffle(idx)
    train_len, test_len = round(data_len*ratios[0]), round(data_len*ratios[1])
    train_idx, test_idx, val_idx = idx[:train_len], idx[train_len:train_len+test_len], idx[train_len+test_len:]

    if spatial_dims == 2:
        train_idx = ImageSlices(train_idx, z)
        test_idx = ImageSlices(test_idx, z)
        val_idx = ImageSlices(val_idx, z)

    return train_idx, test_idx, val_idx

def ImageSlices(dataset_idx, z):
    output = np.ones(len(dataset_idx)*z)

    for i, idx in enumerate(dataset_idx):
        output[z*i:z*(i+1)] = np.arange(z*idx, z*(idx+1))

    return output