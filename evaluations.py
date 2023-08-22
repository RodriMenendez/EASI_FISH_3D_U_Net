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
