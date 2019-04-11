from torch import nn
from src.model.encoder.resnext import AdaptedXception
from src.model.decoder.v3plus_decoder import V3PlusDecoder
from src.model.pyramid_module.aspp import ASPPModule

class DeepLabV3Plus(nn.Module):
    def __int__(self, final_classes=21):
        super().__init__()

        self.encoder = AdaptedXception()
        self.pyramid_module = ASPPModule(in_channels=2048)
        self.decoder = V3PlusDecoder(256, 256, final_classes)

    def forward(self, input_batch):
        dict_output = self.encoder(input_batch)
        pyramid_out = self.pyramid_module(dict_output['final'])
        output = self.decoder(dict_output['interm_batch'], pyramid_out)
        return output

