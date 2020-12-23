import torch


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
    weights = [0, 1, 1, 1, 1, 1, 1]
    weighted_celoss = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights).float().cuda())
    loss = weighted_celoss(outputs, targets)
    # Find where targets isnt 0
    
    # mask = targets != 0
    # output_mask = torch.zeros_like(outputs)
    # # Create mask
    # for i in range(output_mask.shape[1]):
    #     output_mask[:, i, :, :] = targets

    # output_select = outputs[output_mask != 0]
    # targets_select = targets[mask]
    # print("Output select: ", output_select.shape)
    # print("targets_select: ", targets_select.shape)
    # loss = torch.nn.CrossEntropyLoss(output_select, targets_select)
    return loss