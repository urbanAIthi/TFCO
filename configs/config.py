import numpy as np

cnn_autoencoder_config = {
}

transformer_config = {  
    'dim': 1024,
    'temporal_depth': 6,
    'heads': 8,
    'dim_head': 64,
    'mlp_dim': 1024,
    'dropout':0,
    'global_average_pool': 'cls'
}

vit_config = {
    'image_size': 400,
    'patch_size': 50,
    'num_classes': 1000,
    'dim': 1024,
    'depth': 6,
    'heads': 8,
    'mlp_dim': 1024,
    'dropout': 0.25,
    'emb_dropout': 0,
    'channels': 1
}

d3cnn_config = {
    'encoder_active': False,
    'decoder_active': False,
}


autoencoder_config = {
    'pretrain' : True,
    'cnn_depth' : 128,
    'epochs' : [100, 100, 100],
}

default_scheduler_config = {
    'type': 'default',
    'init_lr': 0.0001,
    'step_size': 5000,
    'gamma': 0.9}

cosine_scheduler_config = {
    'type': 'cosine',
    'startup_steps': 5000,
    'min_lr': 1e-6,
}

convlstm_config = {
    'input_dim': 1,
    'hidden_dims': [32,32,32,1],
    'kernel_sizes': (3,3),
    'num_layers': 4,
    'batch_first': True,
    'bias': True,}


special_temporal_decoder_new = { # this is the config used for the spacial temporal decoder training
    'dataset_path': "data",
    'dataset_name_train': 'i3040_newdetector_10p_train',
    'dataset_name_val': 'i3040_newdetector_10p_val',
    'image_size': 512,
    'sequence_len': 5,
    'num_epochs': 400,
    'batch_size': 32,
    'scheduler': {'type': 'default', 'init_lr': 0.0001, 'step_size': 5000, 'gamma': 0.9},

    'criterion': {'types': ['IOU', 'BCE'], 'types_weight': [0.8, 0.2],  'BCE_weight': 1},
    
    'encoder': 'CNN_Encoder',
    'fix_encoder': False,
    'load_encoder': None,

    'temporal': None,
    'load_temporal': None,
    'fix_temporal': False,

    'decoder': 'CNN_Decoder',
    'fix_decoder': False,
    'load_decoder': None,

    'pre_train': True, # if true, the model will be trained in autoencoder style --> input = output
    'pre_train_path': None, #'pre_train_data_100000_0_30_75', # choose spacial dataset for pre-training the autoencoder, if None the normal dataset will be used
    'residual_connection': False, # if true, the last (current) input image of the sequence will be added at the decoder stage
    'sigmoid_factor': 2, # sets a scaling for the sigmoid function after the decoder (e.g. 2 --> sigmoid(x*2))

    'load_complete_model': None,#"trained_spacial_temporal_decoder/i3040_10pct_6-9h_75m_large_30-11_12-29-59/models/model_epoch_22.pth",#'trained_spacial_temporal_decoder/i3040_10pct_6-12h_75m_15-11_07-14-11/models/model_epoch_35.pth',#None#'trained_spacial_temporal_decoder/i3040_10pct_6-12h_75m_14-11_18-36-27/models/model_epoch_395.pth',#None, #'trained_spacial_temporal_decoder/i3040_10pct_6-12h_75m_14-11_12-15-57/models/model_epoch_10.pth',#'trained_spacial_temporal_decoder/i3040_10pct_6-7h_75m_02-11_11-48-42/models/model_epoch_140.pth',#None,#'trained_spacial_temporal_decoder/i3040_10pct_6-7h_75m_31-10_06-06-30/models/model_epoch_255.pth',#'trained_spacial_temporal_decoder/i3040_10pct_6-7h_75m_24-10_16-37-36/models/model_epoch_40.pth',###'trained_spacial_temporal_decoder/i3040_10pct_6-7h_75m_23-10_21-12-12/models/model_epoch_90.pth',####'trained_spacial_temporal_decoder/i3040_20pct_6-7h_75m_20-10_10-36-01/models/best_val_statedict.pth', # load the complete model from a checkopoint (encoder, temporal, decoder). Will override the loading of the sub modules

    # data augmentation settings
    'map_binary': False, # if true, the input will be converted to {0,1} values with threshold 0.5
    'only_occluded': False, # if true, images where there is no temporal potential are removed
    'image_dropout': 0, # fraction of images that are randomly set to zero images
    'image_roation': False, # [int] if value set, the input sequence and the target sequence will be randomly rotated according to a normal distribution (by the same angle)
    'image_shift': False, # [int] if value set, the input sequence and the target sequence will be randomly shifted according to a normal distribution (by the same amount)


    'validation_frequency': 5, # how often the validation is performed (in epochs)
    'save_frequency': 5, # how often the model is saved (in epochs)
}

network_configs = {
    'VIT': vit_config,
    'CNN_Autoencoder': cnn_autoencoder_config,
    'TemporalTransformer': transformer_config,
    '3DCNN': d3cnn_config,
    'ConvLSTM': convlstm_config,
}


