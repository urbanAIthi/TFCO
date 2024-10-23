import time
from vit_pytorch.vit import ViT


import torch
import torch.nn as nn
from torch import nn


import os


class Trans_CNN_Autoencoder(nn.Module):
    def __init__(self, fix_decoder=False):
        super(Trans_CNN_Autoencoder, self).__init__()
        #Initialize pure cnn autoencoder model
        self.cnn_Autoencoder = Autoencoder()

        #Load the pretrained pure autoencoder model
        self.cnn_Autoencoder.load_state_dict(torch.load(os.path.join('models', 'pure_cnn.py')))
        if fix_decoder:
            for param in self.cnn_Autoencoder.parameters():
                param.requires_grad = False

        #Initialize the ViT encoder
        self.vit_encoder = ViT(
                            image_size = 400,
                            patch_size = 50,
                            num_classes = 1000,
                            dim = 1024,
                            depth = 6,
                            heads = 4,
                            mlp_dim = 1024,
                            dropout = 0.25,
                            emb_dropout = 0,
                            channels = 1
                        )

    def forward(self, img):
        #ViT encoder
        x = self.vit_encoder(img)

        #CNN decoder

        x = self.cnn_Autoencoder.fc2(x)
        encoded_fc = self.cnn_Autoencoder.fc_relu2(x)

        # Reshape the encoded tensor
        encoded_fc_reshaped = encoded_fc.view(-1, 16, 50, 50)

        # Decoder
        x = self.cnn_Autoencoder.decoder_conv1(encoded_fc_reshaped)
        x = self.cnn_Autoencoder.decoder_relu1(x)
        x = self.cnn_Autoencoder.decoder_conv2(x)
        x = self.cnn_Autoencoder.decoder_relu2(x)
        x = self.cnn_Autoencoder.decoder_conv3(x)
        decoded = self.cnn_Autoencoder.decoder_sigmoid(x)
        return decoded


class SpacialTemporalDecoderOld(nn.Module):
    def __init__(self, config, fix_decoder=False, load_decoder = False,
                 fix_spacial = False, load_spacial = False):
        super(SpacialTemporalDecoderOld, self).__init__()
        #Initialize pure cnn autoencoder model
        self.cnn_Autoencoder = Autoencoder()

        #Load the pretrained pure autoencoder model
        if load_decoder:
            self.cnn_Autoencoder.load_state_dict(torch.load(os.path.join('models', 'pure_cnn.py')))
        if fix_decoder:
            for param in self.cnn_Autoencoder.parameters():
                param.requires_grad = False

        #Initialiaze the Temporal Transformer
        self.temporal_transformer = TemporalTransformer()

        #Initialize the ViT encoder
        self.vit_encoder = ViT(
                            image_size = 400,
                            patch_size = 50,
                            num_classes = 1000,
                            dim = 1024,
                            depth = 6,
                            heads = 8,
                            mlp_dim = 1024,
                            dropout = 0.25,
                            emb_dropout = 0,
                            channels = 1
                        )


        if load_spacial:
            #load the pretrained vit_encoder_cnn_decoder model
            state_dict = torch.load(os.path.join('models', 'trained_trans_cnn.pth'))
            #extract the vit_encoder state_dict
            desired_keys = [key for key in state_dict.keys() if key.startswith('vit_encoder')]
            filtered_state_dict = {key: state_dict[key] for key in desired_keys}
            filtered_state_dict = {key.replace('vit_encoder.', ''): filtered_state_dict[key] for key in desired_keys}
            # print(filtered_state_dict.keys())
            self.vit_encoder.load_state_dict(filtered_state_dict)
        if fix_spacial:
            for param in self.vit_encoder.parameters():
                param.requires_grad = False

        # create randon torch tensor with shape (sequence_len = 10, batch_size = 32, num_channels = 1, height = 400, width = 400)
        # t = torch.rand(10, 32, 1, 400, 400)
        # reshape to (sequence_len * batch_size, num_channels, height, width)
        # t = t.reshape(320, 1, 400, 400)
        # pass tensor through the special (vit) encoder
        # model = Autoencoder()
        # t = torch.rand(10, 32, 1024)
        # pass through temporal decoder
        # t = torch.rand(32, 1024)
        # pass through cnn decoder to reconstruct image
        # t = torch.rand(32, 1, 400, 400)

    def forward(self, img_sequence):
        t = time.time()
        s, b, c, h, w = img_sequence.size()
        concat_sequence = img_sequence.reshape(s*b, c, h, w)
        #ViT encoder
        x = self.vit_encoder(concat_sequence)
        #print(f'vit_encoder time: {time.time() - t}')

        encoded_sequene = x.reshape(s, b, -1)
        #print(f'encoded_sequene shape: {encoded_sequene.shape}')

        #Temporal Transformer
        temporal_transformed = self.temporal_transformer(encoded_sequene)
        #print(f'temporal_transformed shape: {temporal_transformed.shape}')
        #print(f'temporal_transformer time: {time.time() - t}')
        #CNN decoder

        x = self.cnn_Autoencoder.fc2(temporal_transformed)
        encoded_fc = self.cnn_Autoencoder.fc_relu2(x)

        # Reshape the encoded tensor
        encoded_fc_reshaped = encoded_fc.view(-1, 16, 50, 50)

        # Decoder
        x = self.cnn_Autoencoder.decoder_conv1(encoded_fc_reshaped)
        x = self.cnn_Autoencoder.decoder_relu1(x)
        x = self.cnn_Autoencoder.decoder_conv2(x)
        x = self.cnn_Autoencoder.decoder_relu2(x)
        x = self.cnn_Autoencoder.decoder_conv3(x)
        decoded = self.cnn_Autoencoder.decoder_sigmoid(x)
        #print(f'cnn_decoder time: {time.time() - t}')
        return decoded