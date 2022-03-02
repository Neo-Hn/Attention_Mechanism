import torch
from torch import nn
import math

class Eca_Block(nn.Module):
    def __init__(self, channels, gamma = 2, b = 1):
        super(Eca_Block, self).__init__()
        kernel_size = int(abs((math.log(channels, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        padding = kernel_size // 2
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # 1D卷积是在训练模式下使用的
        self.conv = nn.Conv1d(1, 1, kernel_size = kernel_size, padding = padding, stride = 1, bias = False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
            b: Batch_size
            c: channels
            h: height
            w: width
        """

        b, c, h, w = x.size()
        avg = self.avg_pool(x).view([b, 1, c])
        out = self.conv(avg)
        out = self.sigmoid(out).view([b, c, 1, 1])
        print(out)

        return out * x


model = Eca_Block(channels = 512)
print(model)
inputs = torch.ones([2, 512, 26, 26])
outputs = model(inputs)