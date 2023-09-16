import torch

class DiceLoss():
    def __init__(self):
        pass

    def __call__(self, prediction, target):
        overlap = 2*torch.sum(torch.mul(prediction, target))
        pred_squared = torch.sum(torch.mul(prediction.clone(), prediction.clone()))
        target_squared = torch.sum(torch.mul(target, target))
        union = pred_squared + target_squared
        dice_score = overlap/union
        return 1 - dice_score
    
class SurfaceArea():
    def __init__(self):
        surface_conv = torch.nn.Conv3d(1, 1, 3, padding=1, bias=False)
        surface_conv.weight.requires_grad = False

        z_der_kernel = torch.tensor([[[0, 0, 0],
                              [0, 1, 0],
                              [0, 0, 0]],
  
                             [[0, 1, 0],
                              [1, 0, 1],
                              [0, 1, 0]],
  
                             [[0, 0, 0],
                              [0, 1, 0],
                              [0, 0, 0]]
                            ])

        z_der_kernel = z_der_kernel.to(torch.double)
        surface_conv.weight.data = torch.zeros(surface_conv.weight.shape)
        surface_conv.weight.data[0, 0] = z_der_kernel
        surface_conv.weight.data = surface_conv.weight.data.to(torch.double)

        self.surface_conv = surface_conv

    def __call__(self, sample):
        sample_inverse = torch.abs(sample - 1).to(torch.double)

        surface_area_tensor = self.surface_conv(sample_inverse)
        surface_area_tensor = torch.mul(surface_area_tensor, sample)

        surface_area = torch.sum(surface_area_tensor).item()

        return surface_area
    
def get_surface_area(mask):
    calculate_surface_area = SurfaceArea()
    return calculate_surface_area(mask)

def get_volume(mask, normalize=True):
    norm_value = 1
    if normalize:
        norm_value = torch.prod(torch.tensor(mask.shape)).item()
    
    volume = torch.sum(mask)/norm_value

    return volume

def ConfusionMatrix(preds, labels):
    TP = torch.logical_and((preds == 1), (labels == 1)).float().sum()
    FP = torch.logical_and((preds == 1), (labels == 0)).float().sum()
    TN = torch.logical_and((preds == 0), (labels == 0)).float().sum()
    FN = torch.logical_and((preds == 0), (labels == 1)).float().sum()
    return {'TP':TP, 'TN':TN, 'FP':FP, 'FN':FN}

def accuracy(measures):
    return (measures['TP'] + measures['TN'])/(measures['TP']+measures['TN']+measures['FP']+measures['FN'])

def precision(measures):
    den = measures['TP']+measures['FP']
    output = 0.0 if den == 0 else measures['TP']/den
    return output

def IoU(measures):
    den = measures['TP']+measures['FP']+measures['FN']
    output = 0.0 if den == 0 else measures['TP']/den
    return output

def prediction(output):
    out_channels = output.shape[1]
    if out_channels == 1:
        output[output >= 0.5] = 1
        output[output < 0.5] = 0
    elif out_channels > 1:
        output = torch.argmax(output, dim=1, keepdim=True)

    return output
