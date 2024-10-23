# Define the autoencoder architecture
from sub_models.vit_pytorch import ViT



def init_ViT(configs, network_configs):
    vit_model = ViT(
        image_size=network_configs['VIT']['image_size'],
        patch_size=network_configs['VIT']['patch_size'],
        num_classes=network_configs['VIT']['num_classes'],
        dim=network_configs['VIT']['dim'],
        depth=network_configs['VIT']['depth'],
        heads=network_configs['VIT']['heads'],
        mlp_dim=network_configs['VIT']['mlp_dim'],
        channels=network_configs['VIT']['channels'],
        dropout=network_configs['VIT']['dropout'],
        emb_dropout=network_configs['VIT']['emb_dropout']
    )

    return vit_model
