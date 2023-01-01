import torch
import numpy as np

def loss_fn(X, y):
    return mse_loss(X, y)

def mse_loss(X, y):
    return torch.mean((X[:, 0]-y[:, 0])**2) + torch.mean((X[:, 1]-y[:, 1])**2)

def me_loss(X, y):
    return torch.mean(X[:, 0]-y[:, 0]), torch.mean(X[:, 1]-y[:, 1])

class ScheduledOptim():
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, d_model, n_warmup_steps):
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.init_lr = np.power(d_model, -0.5)

    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        return np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_current_steps += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr

def exists(val):
    return val is not None

def pair(val):
    return (val, val) if not isinstance(val, tuple) else val