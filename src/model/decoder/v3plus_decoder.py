from torch import nn, cat


class V3PlusDecoder(nn.Module):


    def __init__(self, input_low_channels, input_high_channels, final_classes=21):
        super().__init__()

        self.conv1 = nn.Conv2d(input_low_channels, 48, 1)
        self.conv2 = nn.Conv2d(input_high_channels+input_high_channels, 256, 3)
        self.conv3 = nn.Conv2d(256, final_classes, 3)

    def forward(self, low_level_features, aspp_features):
        low_level_features = self.conv1(low_level_features)
        aspp_features = nn.functional.interpolate(aspp_features, scale_factor=4, mode='bilinear')
        combined_features = cat((aspp_features, low_level_features), dim=1)
        combined_features = self.conv2(combined_features)
        combined_features = self.conv3(combined_features)
        combined_features = nn.functional.interpolate(combined_features, scale_factor=4, mode='bilinear')
        return combined_features
