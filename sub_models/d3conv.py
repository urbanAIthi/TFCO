import torch
import torch.nn as nn

class D3CNN(nn.Module):
    def __init__(self, config, network_configs):
        super(D3CNN, self).__init__()
        # Initial Convolution Layers
        self.conv1 = nn.Conv3d(in_channels=6, out_channels=8, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.bn1 = nn.BatchNorm3d(8)
        self.conv2 = nn.Conv3d(in_channels=8, out_channels=16, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(16)
        self.conv3 = nn.Conv3d(in_channels=16, out_channels=16, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.bn3 = nn.BatchNorm3d(16)

        # Transposed Convolution Layers
        self.deconv1 = nn.ConvTranspose3d(in_channels=16, out_channels=16, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.bn4 = nn.BatchNorm3d(16)
        self.deconv2 = nn.ConvTranspose3d(in_channels=16, out_channels=8, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.bn5 = nn.BatchNorm3d(8)
        self.deconv3 = nn.ConvTranspose3d(in_channels=8, out_channels=1, kernel_size=(3, 3, 3), stride=1, padding=1)

    def forward(self, x):
        # Reshape tensor for 3D convolution
        sequence_len, batch_size, channels, height, width = x.shape
        x = x.reshape(batch_size, sequence_len, channels, height, width)
        # Apply initial convolution layers with ReLU and batch normalization
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))

        # Apply transposed convolutions with ReLU and batch normalization to upsample
        x = torch.relu(self.bn4(self.deconv1(x)))
        x = torch.relu(self.bn5(self.deconv2(x)))
        x = self.deconv3(x)

        # Reshape tensor for 2D convolution
        batch_size, channels, depth, height, width = x.size()
        x = x.view(batch_size, channels * depth, height, width)
        return x

if __name__ == "__main__":
    # Define the batch size and input dimensions
    batch_size = 32  # You can change this as needed
    sequence_length = 5
    channels = 1
    height = 400
    width = 400

    # Create a random tensor with the specified dimensions
    # The tensor values are randomly sampled from a normal distribution
    input_tensor = torch.randn(sequence_length, batch_size, channels, height, width)
    # Create an instance of the model
    config = None
    network_configs = None
    model = D3CNN(config, network_configs)
    # Get the output of the model
    out = model(input_tensor)
    # Print the output shape
    print(out.shape)