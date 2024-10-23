import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch
import math

from configs.config import special_temporal_decoder_new as cfg
from configs.config import cosine_scheduler_config as cosine_config
from configs.config import default_scheduler_config as default_config


def create_scheduler(optimizer):
    if cfg['scheduler']['type'] == 'default':
        return lr_scheduler.StepLR(optimizer, step_size=default_config['step_size'], gamma=default_config['gamma'])
    elif cfg['scheduler']['type'] == 'cosine':
        return WarmupCosineAnnealingLR(cosine_config, cfg)  # Adjust these parameters as needed
    else:
        raise NotImplementedError("Scheduler type not implemented")


class WarmupCosineAnnealingLR(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, config, last_epoch=-1):
        self.startup_steps = config.get('startup_steps')
        self.min_lr = config.get('min_lr')
        self.total_steps = cfg.get('num_epochs')  # Total training steps
        self.cosine_annealing_steps = self.total_steps - self.startup_steps
        super(WarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.startup_steps:
            return [base_lr * (self.last_epoch + 1) / self.startup_steps for base_lr in self.base_lrs]
        else:
            return [self.min_lr + (base_lr - self.min_lr) * (1 + math.cos(math.pi * (self.last_epoch - self.startup_steps) / self.cosine_annealing_steps)) / 2 for base_lr in self.base_lrs]
