import torch.nn as nn
import torch

class CNNEncoderOld(nn.Module):
    def __init__(self, dropout_rate=0.25):
        super(CNNEncoder, self).__init__()
        
        # Encoder layers
        self.layers = nn.Sequential(
            nn.Conv2d(1, 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),  # Dropout layer
            nn.Conv2d(2, 1, kernel_size=3, stride=2, padding=1),
        )
        
        self.fc = nn.Sequential(
            nn.Linear(16 * 50 * 50, 1024),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.layers(x)
        x = x.view(x.size(0), -1)  # Flatten tensor
        #x = self.fc(x)
        return x

class CNNDecoderOld(nn.Module):
    def __init__(self, dropout_rate=0.25):
        super(CNNDecoder, self).__init__()

        # ... your existing code

        # Decoder layers
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(2, 8, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Dropout(dropout_rate),  # Dropout layer
            nn.ConvTranspose2d(8, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(dropout_rate),  # Dropout layer
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
        )

        # The merging layers remain unchanged
        self.merge_conv1 = nn.Conv2d(2, 2, kernel_size=3, stride=1, padding=1)
        self.merge_relu1 = nn.ReLU()
        self.merge_conv2 = nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1)
        
    def forward(self, encoded, input, res=False, sigmoid_factor=1):
        #x = self.fc(encoded)
        x = encoded
        
        # Reshape the encoded tensor
        x = x.view(-1, 2, 50, 50)

        # Decoder
        x = self.layers(x)
        if res:
            x = torch.cat((x, input), dim=1)
            x = self.merge_conv1(x)
            x = self.merge_relu1(x)
            decoded = self.merge_conv2(x)
            decoded = nn.Sigmoid()(sigmoid_factor * decoded)
        else:
            decoded = x
            decoded = nn.Sigmoid()(sigmoid_factor * x)

        return decoded


class CNNEncoder(nn.Module):
    def __init__(self):
        super(CNNEncoder, self).__init__()
        self.encoder_layers = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),  # output: [16, 200, 200] (400)
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # output: [32, 100, 100] (400)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),  # output: [64, 50, 50] or [32, 32, 32]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),  # output: [128, 25, 25]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 32 * 32, 1024)
        )

    def forward(self, x):
        return self.encoder_layers(x)


class CNNDecoder(nn.Module):
    def __init__(self):
        super(CNNDecoder, self).__init__()
        self.decoder_layers = nn.Sequential(
            nn.Linear(1024, 32*32*32),
            nn.ReLU(),
            nn.Unflatten(1, (32, 32, 32)),
            nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            #nn.Sigmoid()  # Use Sigmoid for the final layer if the input is normalized between 0 and 1
        )

    def forward(self, x):
        return self.decoder_layers(x)
    

