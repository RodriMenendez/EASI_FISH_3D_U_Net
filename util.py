import numpy as np
import glob

def DataSplit(ratios, data_root, seed, data_len=None):
    np.random.seed(seed)
    if not data_len:
        data_len = len(glob.glob(data_root + 'images/*.tif'))
    idx = np.arange(data_len)
    np.random.shuffle(idx)
    train_len, test_len = round(data_len*ratios[0]), round(data_len*ratios[1])
    train_idx, test_idx, val_idx = idx[:train_len], idx[train_len:train_len+test_len], idx[train_len+test_len:]

    return train_idx, test_idx, val_idx