from torch import nn


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=3, stride=1, padding=0, dilation=1, bias=True):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel, stride, padding,
                                   dilation=dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0,
                                   dilation=1, groups=1, bias=bias)

        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()

    def forward(self, input_batch):
        input_batch = self.depthwise(input_batch)
        input_batch = self.bn(input_batch)
        input_batch = self.relu(input_batch)
        return self.pointwise(input_batch)


class DilationModule(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, padding=0, dilation_rate=1, stride=1):
        super().__init__()
        self.dil_conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation_rate, stride=stride)
        self.batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, input_tensor):
        input_tensor = self.dil_conv(input_tensor)
        return self.batch_norm(input_tensor)
