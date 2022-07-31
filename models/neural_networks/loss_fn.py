import torch.nn as nn
import torch 
def cross_entropy_loss_binary():
    # There is only a single output variable, so use th 
    # binary cross entropy loss 
    return nn.BCELoss()

def cross_entropy_loss_2d(weights=torch.tensor([0.97,0.03])):
    # There is only a single output variable, so use th 
    # binary cross entropy loss 
    return nn.CrossEntropyLoss(weight=weights)

def mean_squared_error_loss():
    return nn.MSELoss()

def weighted_binary_cross_entropy(output, target, weights=None):
    '''
        Adds weights to different classes, 
    '''
    if weights is not None:
        assert len(weights) == 2
        
        loss = weights[1] * (target * torch.log(output)) + \
               weights[0] * ((1 - target) * torch.log(1 - output))
    else:
        loss = target * torch.log(output) + (1 - target) * torch.log(1 - output)

    return torch.neg(torch.mean(loss))