import torch
from torch import nn

class SE_Block(nn.Module):
    def __init__(self, in_channels, ratio = 16):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc       = nn.Sequential(
            nn.Linear(in_channels, in_channels // ratio, bias = False),
            nn.ReLU(),
            nn.Linear(in_channels // ratio, in_channels, bias = False),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
            b: Batch_size
            c: channels
            h: height
            w: width
        """
        b, c, h, w = x.size()

        # after avg_pool, the x is become: b, c, h, w -> b, c, 1, 1
        avg = self.avg_pool(x).view([b, c])

        # b, c -> (b, c) // ratio -> b, c -> b, c, 1, 1
        fc  = self.fc(avg).view([b, c, 1, 1])

        return x * fc

model = SE_Block(512)
print(model)
inputs = torch.ones([2, 512, 26, 26])
outputs = model(inputs)

