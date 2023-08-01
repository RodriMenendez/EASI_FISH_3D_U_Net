import torch
import torch.nn.functional as F

class Resize3D():
    def __init__(self, size, mode="bilinear"):
        self.size = size
        self.mode = mode

    def __call__(self, sample):
        d = torch.linspace(-1, 1, self.size[0])
        h = torch.linspace(-1, 1, self.size[1])
        w = torch.linspace(-1, 1, self.size[2])
        meshz, meshy, meshx = torch.meshgrid((d, h, w), indexing='ij')
        grid = torch.stack((meshx, meshy, meshz), 3)
        grid = grid.unsqueeze(0).float() # add batch dim


        out = F.grid_sample(sample.unsqueeze(0).float(), grid, mode = self.mode, align_corners=True).squeeze(0)

        return out
    
class BinaryReplace():
    def __init__(self, limit):
        self.limit = limit
    
    def __call__(self, sample):
        sample[sample <= self.limit] = 0
        sample[sample > self.limit] = 1

        return sample