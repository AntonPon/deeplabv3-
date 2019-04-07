from torch import nn
from src.model.utils.additional_modules import SeparableConv2d, DilationModule


class BuildingBlock(nn.Module):
    def __init__(self, in_channels, out_channels=(128, 128, 128), kernel=3, reduce_block=True, skip_connection=True):
        super().__init__()
        if type(out_channels) != tuple and len(out_channels) != 3:
            raise ValueError('the number of output channels != 3 or != 1 ')
        else:
            out_channels = [out_channels] * 3
        stride = 1
        if reduce_block:
            stride = 2

        self.sepConv1 = SeparableConv2d(in_channels, out_channels[0], kernel=kernel)
        self.bn1 = nn.BatchNorm2d(out_channels[0])
        self.sepConv2 = SeparableConv2d(out_channels[0], out_channels[1], kernel=kernel)
        self.bn2 = nn.BatchNorm2d(out_channels[1])
        self.sepConv3 = SeparableConv2d(out_channels[1], out_channels[2], kernel=kernel, stride=stride)
        self.bn3 = nn.BatchNorm2d(out_channels[2])
        self.relu = nn.ReLU()

        self.skip_connection = None
        if skip_connection:
            self.skip_connection = nn.Conv2d(in_channels, out_channels[2], kernel_size=1, stride=2)

    def forward(self, input_batch):
        skip_connection = self.skip_connection(input_batch)
        input_batch = self.sepConv1(input_batch)
        input_batch = self.bn1(input_batch)
        input_batch = self.relu(input_batch)

        input_batch = self.sepConv2(input_batch)
        input_batch = self.bn2(input_batch)
        input_batch = self.relu(input_batch)

        input_batch = self.sepConv3(input_batch)
        input_batch = self.bn3(input_batch)
        if self.skip_connection is not None:
            return self.relu(input_batch + skip_connection)
        else:
            return self.relu(input_batch)

class EntryFlow(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()

        self.conv_module1 = DilationModule(in_channels, 32, kernel_size=3, stride=2)
        self.relu = nn.ReLU()
        self.conv_module2 = DilationModule(32, 64, 3)
        self.conv_block1 = BuildingBlock(64, 128, 3)
        self.conv_block2 = BuildingBlock(128, 256, 3)
        self.conv_block3 = BuildingBlock(256, 728, 3)

    def forward(self, input_batch):
        input_batch = self.conv_module1(input_batch)
        input_batch = self.relu(input_batch)
        input_batch = self.conv_module2(input_batch)
        input_batch = self.relu(input_batch)
        input_batch = self.conv_block1(input_batch)
        input_batch = self.conv_block2(input_batch)
        input_batch = self.conv_block3(input_batch)
        return input_batch

class MiddleFlow(nn.Module):
    def __init__(self, in_channels=728, total_blocks=16):
        super().__init__()

        self.block = BuildingBlock(in_channels, 728, 3, reduce_block=False)
        self.blocks = nn.ModuleList([BuildingBlock(728, 728, 3, reduce_block=False) for i in range(total_blocks-1)])

    def forward(self, input_batch):
        input_batch = self.block(input_batch)
        for b_block in self.blocks:
            input_batch = b_block(input_batch)
        return input_batch


class ExitFlow(nn.Module):
    def __init__(self, in_channels=728):
        super().__init__()

        self.conv_block1 = BuildingBlock(in_channels, (728, 1024, 1024), 3)
        self.conv_block2 = BuildingBlock(1024, (1536, 1536, 2048), 3, reduce_block=False, skip_connection=False)

    def forward(self, input_batch):
        input_batch = self.conv_block1(input_batch)
        return self.conv_block2(input_batch)



class AdaptedXception(nn.Module):
    def __init__(self, in_channels=3, total_middle_blocks=16):
        super().__init__()

        self.entry = EntryFlow(in_channels)
        self.middle = MiddleFlow(total_blocks=total_middle_blocks)
        self.exit = ExitFlow()

    def forward(self, input_batch):
        result = self.entry(input_batch)
        result = self.middle(result)
        return self.exit(result)
