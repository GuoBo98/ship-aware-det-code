import torch
import torch.nn as nn 

batch_norm1 = nn.BatchNorm2d(64) # NCHW ---> C
input = torch.randn(size = (1,64,512,512))
output = batch_norm1(input)

layer_norm = nn.BatchNorm2d(input.shape[1:])
output = layer_norm(input)