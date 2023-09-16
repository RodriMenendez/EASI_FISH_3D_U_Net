import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms

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
    
class MaxPool():
    def __init__(self, shape):
        self.shape = shape

    def __call__(self, sample):
        sample_shape = torch.tensor(sample.squeeze(0).shape)
        maxpool_shape = torch.div(sample_shape, torch.tensor(self.shape), rounding_mode='floor')

        maxpool_shape[maxpool_shape < 1] = 1
        maxpool_shape = maxpool_shape.tolist()

        maxpool = nn.MaxPool3d(maxpool_shape)
        sample = maxpool(sample)

        return sample
    
class BoundBox():
    def __init__(self):
        pass

    def get_bounding_box(self, input_tensor):
        if len(input_tensor.shape) > 3:
            input_tensor = input_tensor.squeeze(0)
            return self.get_bounding_box(input_tensor)

        z_max_projection = torch.argmax(input_tensor, axis=0)
        z_min = z_max_projection[z_max_projection != 0].min()
        z_max_projection_reverse = torch.argmax(torch.flip(input_tensor, dims=(0,)), axis=0)
        z_max = input_tensor.shape[0] - (z_max_projection_reverse[z_max_projection_reverse != 0].min() + 1)

        y_max_projection = torch.argmax(input_tensor, axis=1)
        y_min = y_max_projection[y_max_projection != 0].min()
        y_max_projection_reverse = torch.argmax(torch.flip(input_tensor, dims=(1,)), axis=1)
        y_max = input_tensor.shape[1] - (y_max_projection_reverse[y_max_projection_reverse != 0].min() + 1)

        x_max_projection = torch.argmax(input_tensor, axis=2)
        x_min = x_max_projection[x_max_projection != 0].min()
        x_max_projection_reverse = torch.argmax(torch.flip(input_tensor, dims=(2,)), axis=2)
        x_max = input_tensor.shape[2] - (x_max_projection_reverse[x_max_projection_reverse != 0].min() + 1)

        input_tensor = input_tensor[z_min:z_max+1, y_min:y_max+1, x_min:x_max+1]
        return input_tensor

    def __call__(self, sample):
        return self.get_bounding_box(sample)
    
class CustomTransformation():
    def __init__(self, transformations, shape, mask=False):
        transforms_list = []
        
        if 'maxpool' in transformations:
            transforms_list.append(MaxPool(shape))

        if 'bilinear' in transformations:
            transforms_list.append(Resize3D(shape, 'bilinear'))
        elif 'nearest' in transformations:
            transforms_list.append(Resize3D(shape, 'nearest'))
        else:
            TypeError('Must have a resize transformation')

        if mask:
            transforms_list.append(BinaryReplace(0.0))

        self.transforms = transforms.Compose(transforms_list)

    def __call__(self, sample):
        return self.transforms(sample)