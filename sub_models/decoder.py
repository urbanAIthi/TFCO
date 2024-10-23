import torch.nn as nn
from torch import nn
import torch


class Reshape(nn.Module):
    def forward(self, x):
        return x.view(-1, 16, 50, 50)


class Autoencoder(nn.Module):
    def __init__(self, config, network_configs):
        super(Autoencoder, self).__init__()

        # Encoder layers
        self.encoder_conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        self.encoder_relu1 = nn.ReLU()
        self.encoder_conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)
        self.encoder_relu2 = nn.ReLU()
        self.encoder_conv3 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)
        self.encoder_relu3 = nn.ReLU()
        #
        # # Fully connected layers
        self.fc1 = nn.Linear(16 * 50 * 50, 1024)
        self.fc_relu1 = nn.ReLU()
        self.fc2 = nn.Linear(1024, 16 * 50 * 50)
        self.fc_relu2 = nn.ReLU()
        #
        # # Decoder layers
        self.decoder_conv1 = nn.ConvTranspose2d(16, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.decoder_relu1 = nn.ReLU()
        self.decoder_conv2 = nn.ConvTranspose2d(16, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.decoder_relu2 = nn.ReLU()
        self.decoder_conv3 = nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.merge_conv1 = nn.Conv2d(2, 2, kernel_size=3, stride=1, padding=1)
        self.merge_relu1 = nn.ReLU()
        self.merge_conv2 = nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1)
        self.decoder_sigmoid = nn.Sigmoid()
    
    def forward(self, encoded, input, res=False, sigmoid_factor=1):
        x = self.fc2(encoded)
        encoded_fc = self.fc_relu2(x)

        # Reshape the encoded tensor
        encoded_fc_reshaped = encoded_fc.view(-1, 16, 50, 50)

        # Decoder
        x = self.decoder_conv1(encoded_fc_reshaped)
        x = self.decoder_relu1(x)
        x = self.decoder_conv2(x)
        x = self.decoder_relu2(x)
        x = self.decoder_conv3(x)
        if res:
            if False:
                # plot the x image 
                image = tensor.squeeze().cpu().numpy()
                plt.imshow(image, cmap='gray')
                plt.savefig(f'test.png')
            x = torch.cat((x, input), dim=1)
            x = self.merge_conv1(x)
            x = self.merge_relu1(x)
            decoded = self.merge_conv2(x)
            decoded = self.decoder_sigmoid(sigmoid_factor*decoded)
        else:
            decoded = self.decoder_sigmoid(sigmoid_factor*x)

        return decoded


    def forward_autoencoder(self, x):
        # Encoder
        x = self.encoder_conv1(x)
        x = self.encoder_relu1(x)
        x = self.encoder_conv2(x)
        x = self.encoder_relu2(x)
        x = self.encoder_conv3(x)
        encoded = self.encoder_relu3(x)

        # Flatten the encoded tensor
        x = torch.flatten(encoded, start_dim=1)

        # Fully connected layers
        x = self.fc1(x)
        x = self.fc_relu1(x)
        x = self.fc2(x)
        encoded_fc = self.fc_relu2(x)

        # Reshape the encoded tensor
        encoded_fc_reshaped = encoded_fc.view(-1, 16, 50, 50)

        #Decoder
        x = self.decoder_conv1(encoded_fc_reshaped)
        x = self.decoder_relu1(x)
        x = self.decoder_conv2(x)
        x = self.decoder_relu2(x)
        x = self.decoder_conv3(x)
        decoded = self.decoder_sigmoid(x)

        return decoded

    def forward_pre1(self, x):
        # Encoder
        x = self.encoder_conv1(x)
        x = self.encoder_relu1(x)

        x = self.decoder_conv3(x)
        decoded = self.decoder_sigmoid(x)

        return decoded

    def forward_pre2(self, x):
        # Encoder
        x = self.encoder_conv1(x)
        x = self.encoder_relu1(x)
        x = self.encoder_conv2(x)
        x = self.encoder_relu2(x)

        x = self.decoder_conv2(x)
        x = self.decoder_relu2(x)
        x = self.decoder_conv3(x)
        decoded = self.decoder_sigmoid(x)

        return decoded

    def forward_pre3(self, x):
        # Encoder
        x = self.encoder_conv1(x)
        x = self.encoder_relu1(x)
        x = self.encoder_conv2(x)
        x = self.encoder_relu2(x)
        x = self.encoder_conv3(x)
        encoded = self.encoder_relu3(x)

        x = self.decoder_conv1(encoded)
        x = self.decoder_relu1(x)
        x = self.decoder_conv2(x)
        x = self.decoder_relu2(x)
        x = self.decoder_conv3(x)
        decoded = self.decoder_sigmoid(x)

        return decoded

    def set_pretrain_list(self):
        self.pretrain_list = [self.forward_pre1, self.forward_pre2, self.forward_pre3, self.forward]
        self.pretrain_settings = [{'lr': 0.01, 'epochs': 100, 'scheduler_step_size': 10, 'scheduler_gamma': 0.9},
                                  {'lr': 0.01, 'epochs': 100, 'scheduler_step_size': 10, 'scheduler_gamma': 0.9},
                                  {'lr': 0.001, 'epochs': 100, 'scheduler_step_size': 10, 'scheduler_gamma': 0.9},
                                  {'lr': 0.001, 'epochs': 500, 'scheduler_step_size': 50, 'scheduler_gamma': 0.9}]
        self.pretrain_state_dicts = [None,
                                     'pretrain1',
                                     'pretrain2',
                                     'pretrain3',
                                     'full']
        return self.pretrain_list