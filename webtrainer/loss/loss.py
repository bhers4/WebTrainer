import torch
import numpy as np


supported_losses = ['BCELoss', 'CrossEntropyLoss', "NLLLoss", "CustomCrossEntropyLoss"]

def load_loss(loss_info, custom_loss=None):
    loss_name = loss_info['name']
    if loss_name in supported_losses:
        print("Loading supported loss func")
        if loss_name == "BCELoss":
            loss = torch.nn.BCELoss()
        elif loss_name == "CrossEntropyLoss":
            loss = torch.nn.CrossEntropyLoss()
        elif loss_name == "NLLLoss":
            loss = torch.nn.NLLLoss()
        elif loss_name == "CustomCrossEntropyLoss":
            loss = CustomCrossEntropyLoss
        else:
            print("Unknown Loss... Defaulting to CrossEntropy")
            loss = torch.nn.CrossEntropyLoss()
    else:
        print("Custom Loss")
        loss = custom_loss
    return loss


def CustomCrossEntropyLoss(outputs, targets):
    # Try weights 0
    weights = np.ones((outputs.shape[1],))
    weights[0] = 0
    weighted_celoss = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights).float().cuda())
    loss = weighted_celoss(outputs, targets)
    return loss