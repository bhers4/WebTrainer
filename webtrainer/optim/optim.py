from torch import optim
from torch.optim import lr_scheduler

supported_optims = ['Adam', 'SGD', 'Adagrad', 'RMSProp']

def load_optim(optimizer, model, custom_optim=None, epochs=None):
    run_optim = ""
    optim_name = optimizer['name']
    lr = optimizer['lr']
    if 'lr_min' in optimizer.keys():
        lr_min = optimizer['lr_min']
    else:
        lr_min = lr*0.01  # Hardcoded for now but if we dont specify lets go from lr to 1/100th of lr

    if optim_name in supported_optims:
        if optim_name == "Adam":
            run_optim = optim.Adam(model.parameters(), lr=lr)
        elif optim_name == "SGD":
            run_optim = optim.SGD(model.parameters(), lr=lr)
        elif optim_name == "Adagrad":
            run_optim = optim.Adagrad(model.parameters(), lr=lr)
        elif optim_name == "RMSProp":
            run_optim = optim.RMSProp(model.parameters(), lr=lr)
        else:
            print("Unknown optim defaulting to...Adam")
            run_optim = optim.Adam(model.parameters(), lr=lr)
    else:
        run_optim = custom_optim
    # Now if we specified a scheduler use it
    if 'scheduler' in optimizer.keys() and 'scheduler_type' in optimizer.keys():
        use_scheduler = optimizer['scheduler']
        scheduler_type = optimizer['scheduler_type']
        if use_scheduler:
            if scheduler_type == "linear":
                step_size = (lr - lr_min) / epochs
                lr_lambda = lambda epoch: epoch - step_size
                lr_optim = CustomLRScheduler(run_optim, lr_lambda)
            elif scheduler_type == "multiplicative":
                if 'multiplier' in optimizer.keys():
                    multiplier = optimizer['multiplier']
                else:
                    multiplier = 0.95  # Default to this
                # torch.optim.lr_scheduler.MultiplicativeLR doesn't seem to be in torch 1.5.0 so make the same function
                lr_lambda = lambda epoch: epoch * multiplier
                lr_optim = CustomLRScheduler(run_optim, lr_lambda)
            else:
                if 'multiplier' in optimizer.keys():
                    multiplier = optimizer['multiplier']
                else:
                    multiplier = 0.95  # Default to this
                # torch.optim.lr_scheduler.MultiplicativeLR doesn't seem to be in torch 1.5.0 so make the same function
                lr_lambda = lambda epoch: epoch * multiplier
                lr_optim = CustomLRScheduler(run_optim, lr_lambda)
            return lr_optim
        else:
            print("Use scheduler is false")
    return run_optim


class CustomLRScheduler(object):

    def __init__(self, optim, lr_lambda):
        self.optim = optim
        self.lr_lambda = lr_lambda
        self.initial_lr = self.optim.param_groups[0]['lr']
        return

    def zero_grad(self):
        self.optim.zero_grad()
        return

    def set_optim_lrs(self, epoch=1):
        # for group in self.optim.param_groups:
        for i, group in enumerate(self.optim.param_groups, 0):
            # Calculate new lr by chaining together lambdas
            new_lr = self.initial_lr
            for j in range(epoch):
                new_lr = self.lr_lambda(new_lr)
            self.optim.param_groups[i]['lr'] = new_lr
        return

    def print_lr(self):
        lrs = []
        for group in self.optim.param_groups:
            lrs.append(group['lr'])
        print("Lrs: ", lrs)
        return

    def step(self, epoch=None):
        # We don't want to step every iter, we want to step every epoch
        if epoch:
            self.optim.step()  # Start by stepping
            self.set_optim_lrs(epoch=epoch)
        else:
            self.optim.step()
        return


