import torch
import torch.nn as nn
#https://chat.openai.com/share/f6d2bb12-2d6a-41bc-a4d6-e6714a422225

import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from sub_models.temporal import TemporalTransformer
import math

# Define Positional Encoding


if __name__ == "__main__":
    model = TemporalTransformer()
    #sequence_len, batch_size, feature_size
    src = torch.rand(10, 64, 1024)
    out = model(src)
    print(out.shape)