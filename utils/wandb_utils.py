import wandb
from configs.config import *

wandb_config = {
    'cnn_autoencoder': cnn_autoencoder_config,
    'transformer': transformer_config,
    'autoencoder': autoencoder_config,
    'scheduler': {
        'default': default_scheduler_config,
        'cosine': cosine_scheduler_config
    },
    'special_temporal_decoder': special_temporal_decoder_new
}



def start_wandb(cfg, filename):
    wandb_mode = ""
    while wandb_mode not in ["online", "offline", "disabled"]:  
        wandb_mode = input('wandb mode (online, offline, disabled): ')
    print(f'wandb mode: {wandb_mode}')
    wandb.init(project="solving_occlusion_spacial_temporal_decoder", mode=wandb_mode, name=filename)
    wandb.config.update(wandb_config)


