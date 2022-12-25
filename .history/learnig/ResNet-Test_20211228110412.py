from mmdet.models import ResNet
import torch
self = ResNet(50)
inputs = torch.rand(1, 3, 32, 32)
level_outputs = self.forward(inputs)
for level_out in level_outputs:
    print(tuple(level_out.shape))
print('hello')