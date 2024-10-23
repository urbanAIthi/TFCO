import torch
import torch.nn as nn
from sub_models.autoencoder import CNNDecoder, CNNEncoder
from sub_models.decoder import Autoencoder
from sub_models.encoder import init_ViT
from sub_models.temporal import TemporalTransformerOld, TemporalConv, TemporalLSTM
from sub_models.conv_lstm import ConvLSTM
from sub_models.d3conv import D3CNN

SUBNETWORKS = {
    'Encoder': {
        'CNN_Encoder': CNNEncoder,
        'VIT': init_ViT,
    },
    'TemporalModule': {
        'Transformer': {"model_class": TemporalTransformerOld, "encoder_active": True, "decoder_active": True},
        'Conv': {"model_class": TemporalConv, "encoder_active": False, "decoder_active": False},
        'LSTM': {"model_class": TemporalLSTM, "encoder_active": True, "decoder_active": True},
        '3DCNN': {"model_class": D3CNN, "encoder_active": False, "decoder_active": False},
        'ConvLSTM': {"model_class": ConvLSTM, "encoder_active": False, "decoder_active": False},
    },
    'Decoder': {
        'CNN_Autoencoder': Autoencoder,
        'CNN_Decoder': CNNDecoder,
    }
}

class SpacialTemporalDecoder(nn.Module):
    def __init__(self, config, network_configs): 
        super(SpacialTemporalDecoder, self).__init__()
        if config['temporal'] is not None:
            self.temporal_module = SUBNETWORKS['TemporalModule'][config['temporal']]['model_class'](config, network_configs)
            self.encoder_active = SUBNETWORKS['TemporalModule'][config['temporal']]['encoder_active']
            self.decoder_active = SUBNETWORKS['TemporalModule'][config['temporal']]['decoder_active']
        else:
            self.temporal_module = None
            self.encoder_active = True
            self.decoder_active = True
            
        if self.encoder_active:
            self.encoder = SUBNETWORKS['Encoder'][config['encoder']]()
        if self.decoder_active:
            self.decoder = SUBNETWORKS['Decoder'][config['decoder']]()

        self.load_pretrained(config)
        self.fix_subnetworks(config)

        self.pre_train = config['pre_train']
        self.sequence_len = config['sequence_len']
        self.res_con = config['residual_connection']
        self.sigmoid_factor = config['sigmoid_factor']
        #assert not (self.pre_train and self.sequence_len != 0), 'sequence_len must be 0 if pre_train is True'
        self.temporal_active = config['temporal'] is not None


    def forward(self, x):
        residual = x[-1,...]
        # input will be of shape (sequence_length, batch_size, channels, height, width)
        s, b, c, h, w = x.shape
        if self.encoder_active:
            x = x.reshape(s*b, c, h, w)
            x = self.encoder(x)
            x = x.reshape(s, b, -1)
        if self.temporal_active:
            x = self.temporal_module(x)
        else:
            x = x.squeeze(dim=1)
        if self.decoder_active:
            x = self.decoder(x)#, residual, sigmoid_factor=2,res=self.res_con)
        return x
    
    def load_pretrained(self, config):
        if config['load_encoder'] is not None:
            self.encoder.load_state_dict(torch.load(config['load_encoder']), strict=False)
            print('loaded encoder')
        
        if config['load_temporal'] is not None:
            self.temporal_module.load_state_dict(torch.load(config['load_temporal']), strict=False)
            print('loaded temporal')
        
        if config['load_decoder'] is not None:
            self.decoder.load_state_dict(torch.load(config['load_decoder']), strict=False)
            print('loaded decoder')
    
    def fix_subnetworks(self, config):
        if config['fix_encoder']:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        if config['fix_temporal']:
            for param in self.temporal_module.parameters():
                param.requires_grad = False
        
        if config['fix_decoder']:
            for param in self.decoder.parameters():
                param.requires_grad = False