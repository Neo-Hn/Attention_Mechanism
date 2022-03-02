import torch
from torch import nn

# 通道注意力部分（CBAM_Block）
class channel_attention(nn.Module):
    def __init__(self, in_channels, ratio = 16):
        super(channel_attention, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc       = nn.Sequential(
            nn.Linear(in_channels, in_channels // ratio, bias = False),
            nn.ReLU(),
            nn.Linear(in_channels // ratio, in_channels, bias = False),
        )
        self.sigmoid  = nn.Sigmoid()

    def forward(self, x):
        """
            b: Batch_size
            c: channels
            h: height
            w: width
        """
        b, c, h, w = x.size()

        # the shape of x in before -> (b, c, h, w)
        # after avg_pool and max_pool, the shape of x will become -> (b, c, 1, 1)
        avg_pool_out = self.avg_pool(x).view([b, c])
        max_pool_out = self.max_pool(x).view([b, c])

        avg_fc_out = self.fc(avg_pool_out)
        max_fc_out = self.fc(max_pool_out)

        out = avg_fc_out + max_fc_out

        # because we need to multiply out by x,
        # so we should supplement out's dimensionality in height and width
        out = self.sigmoid(out).view([b, c, 1, 1])

        return x * out


# 空间注意力部分（CBAM_Block）
class spatial_attention(nn.Module):
    def __init__(self, kernel_size = 7):
        super(spatial_attention, self).__init__()

        padding = kernel_size // 2
        self.conv     = nn.Conv2d(
            in_channels= 2,
            out_channels= 1,
            kernel_size = kernel_size,
            stride = 1,
            padding = padding,
            bias = False
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_pool_out, _ = torch.max(x, dim = 1, keepdim = True)
        avg_pool_out = torch.mean(x, dim = 1, keepdim = True)
        pool_out = torch.cat([max_pool_out, avg_pool_out], dim = 1)

        out = self.conv(pool_out)

        out = self.sigmoid(out)

        return out * x


class Cbam(nn.Module):
    def __init__(self, channels, ratio = 16, kernel_size = 7):
        super(Cbam, self).__init__()
        self.channel_attention = channel_attention(in_channels = channels, ratio = ratio)
        self.spatial_attention = spatial_attention(kernel_size = kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)

        return x


model = Cbam(512)
print(model)
inputs = torch.ones([2, 512, 26, 26])
outputs = model(inputs)