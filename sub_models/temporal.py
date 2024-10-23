# Define the Transformer Model
import torch
import torch.nn as nn
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from utils.transformer_utils import exists, PositionalEncoding, FeedForward, Attention
from einops import rearrange, repeat, reduce


class TemporalTransformerOld(nn.Module):
    def __init__(self, config, network_configs):
        super(TemporalTransformerOld, self).__init__()
        feature_size=1024
        num_layers=6
        num_heads=4
        dropout=0.1
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(feature_size, dropout)
        encoder_layers = nn.TransformerEncoderLayer(feature_size, num_heads, feature_size*4, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.decoder = nn.Linear(feature_size, feature_size)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return output[-1]

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

class TemporalTransformer(nn.Module):
    def __init__(self, config, network_configs):
        super(TemporalTransformer, self).__init__()
        dim = network_configs['TemporalTransformer']['dim']
        temporal_depth = network_configs['TemporalTransformer']['temporal_depth']
        heads = network_configs['TemporalTransformer']['heads']
        dim_head = network_configs['TemporalTransformer']['dim_head']
        mlp_dim = network_configs['TemporalTransformer']['mlp_dim']
        dropout = network_configs['TemporalTransformer']['dropout']

        self.global_average_pool = network_configs['TemporalTransformer']['global_average_pool']

        self.temporal_transformer = Transformer(dim, temporal_depth, heads, dim_head, mlp_dim, dropout)
        self.temporal_cls_token = nn.Parameter(torch.randn(1, 1, dim)) if not self.global_average_pool else None

        self.pos_encoding = PositionalEncoding(dim, dropout)
    
    def forward(self, x):
        b, s, f = x.shape

        # apply positional encoding
        x = self.pos_encoding(x)

        # append temporal CLS tokens

        if exists(self.temporal_cls_token):
            temporal_cls_tokens = repeat(self.temporal_cls_token, '1 1 d-> b 1 d', b = b)

            x = torch.cat((temporal_cls_tokens, x), dim = 1)

        # attend across time

        x = self.temporal_transformer(x)

        # excise out temporal cls token or average pool

        x = x[:, 0] if not self.global_average_pool else reduce(x, 'b f d -> b d', 'mean')

        return x
    


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

class TemporalConv(nn.Module):
    def __init__(self, config, network_configs):
        super(TemporalConv, self).__init__()
        self.depth = config['sequence_len'] + 1

        self.conv1_seq = nn.Conv2d(in_channels=self.depth, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  # Batch Norm for conv1
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)  # Batch Norm for conv2
        
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)  # Batch Norm for conv3
        
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=10, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(10)  # Batch Norm for conv4
        
        self.conv5 = nn.Conv2d(in_channels=10, out_channels=1, kernel_size=3, padding=1)
        
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        s, b, c, h, w = x.shape
        x = rearrange(x, 's b c h w -> b s c h w')
        x = x.reshape(b, s*c, h, w)
        x = self.relu(self.bn1(self.conv1_seq(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.conv5(x) 
        
        # map output to values between 0 and 1
        return x

class TemporalLSTM(nn.Module):
    def __init__(self, config, network_configs):
        super(TemporalLSTM, self).__init__()
        input_dim = 1024
        hidden_dim = 1024
        output_dim = 1024
        num_layers = 2
        self.hidden_dim = hidden_dim

        # Define the LSTM layer with batch_first=False
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=False)

        # Define the output layer
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.lstm.num_layers, x.size(1), self.hidden_dim).to(x.device)

        # Initialize cell state
        c0 = torch.zeros(self.lstm.num_layers, x.size(1), self.hidden_dim).to(x.device)

        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Index hidden state of last time step
        out = self.linear(out[-1, :, :])
        return out


class Conv3DAutoencoder(nn.Module):
    def __init__(self):
        super(Conv3DAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, stride=1, padding=1),  # Output size: 16, N, 400, 400
            nn.ReLU(True),
            nn.MaxPool3d(2, stride=2),  # Output size: 16, N/2, 200, 200
            nn.Conv3d(16, 8, kernel_size=3, stride=1, padding=1),  # Output size: 8, N/2, 200, 200
            nn.ReLU(True),
            nn.MaxPool3d(2, stride=2)  # Output size: 8, N/4, 100, 100
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(8, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output size: 16, N/2, 200, 200
            nn.ReLU(True),
            nn.ConvTranspose3d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1)  # Output size: 1, N, 400, 400
        )
        # create dummy conv that takes image of 1,400,400 and outputs 1,400,400
        self.dummy_conv = nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        s, b, c, h, w = x.shape
        x = x.reshape(b, c, s, h, w)
        x = self.encoder(x)
        x = self.decoder(x)
        # Sum along the sequence dimension to produce a single image
        #x = x.sum(dim=2, keepdim=False)  # Output size: B, 1, 400, 400

        x = x[:,:,-1,...]
        #x = self.dummy_conv(x)
        return x
