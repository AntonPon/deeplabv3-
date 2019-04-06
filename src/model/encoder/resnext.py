from torch import nn


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=3, stride=1, padding=0, dilation=1, bias=True):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel, stride, padding,
                                   dilation=dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel=1, stride=1, padding=0,
                                   dilation=1, groups=1, bias=bias)

        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()

    def forward(self, input_batch):
        input_batch = self.depthwise(input_batch)
        input_batch = self.bn(input_batch)
        input_batch = self.relu(input_batch)
        return self.pointwise(input_batch)


class BuildingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=3):
        super().__init__()
        self.sepConv1 = SeparableConv2d(in_channels, out_channels, kernel=kernel)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.sepConv2 = SeparableConv2d(out_channels, out_channels, kernel=kernel)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.sepConv3 = SeparableConv2d(out_channels, out_channels, kernel=kernel, stride=2)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, input_batch):
        input_batch = self.sepConv1(input_batch)
        input_batch = self.bn1(input_batch)
        input_batch = self.relu(input_batch)

        input_batch = self.sepConv2(input_batch)
        input_batch = self.bn2(input_batch)
        input_batch = self.relu(input_batch)

        input_batch = self.sepConv3(input_batch)
        input_batch = self.bn3(input_batch)
        return self.relu(input_batch)


class EntryFlow(nn.Module):
    def __init__(self):
        super().__init__()



class MiddleFlow(nn.Module):
    pass


class ExitFlow(nn.Module):
    pass


class AdaptedXception(nn.Module):
    pass