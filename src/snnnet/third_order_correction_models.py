from torch import nn
from snnnet.layers import SnConv2
from snnnet.third_order_models import calc_num_expand_channels
from snnnet.third_order_layers import SnConv3, SnConv3to1, SnConv3to2, expand_2_to_3


class ThirdOrderCorrectionEncoder(nn.Module):
    def __init__(self, old_encoder_modules, new_channels_encoder,
                 last_out_channel=100, num_latent=7, num_node_channels=22,
                 nonlinearity=None, expand_type='individual', outer_type='individual',
                 pdim1=1):
        super().__init__()
        self.old_encoder_modules = old_encoder_modules
        self.encoder_polish = SnConv2(last_out_channel,
                                      new_channels_encoder[0], pdim1=pdim1)

        in_channel_i = calc_num_expand_channels(new_channels_encoder[0], expand_type)
        self.new_encoder_layers = nn.ModuleList()
        for out_channel_i in new_channels_encoder[1:]:
            self.new_encoder_layers.append(SnConv3(in_channel_i, out_channel_i))
            in_channel_i = out_channel_i
        self.encode_to_latent = SnConv3to1(in_channel_i, 2 * num_latent,
                                           pdim1=pdim1)

        if nonlinearity is None:
            nonlinearity = nn.ReLU
        self.nonlinearity = nonlinearity()

        self.expand_type = expand_type
        self.outer_type = outer_type
        self.pdim1 = pdim1

    def forward(self, x, x_node=None, mask=None):
        for module in self.old_encoder_modules:
            x = module(x, x_node=x_node)
            x_node = None
            x = self.nonlinearity(x)
        x = self.encoder_polish(x)

        x = expand_2_to_3(x, mode=self.expand_type)

        for module in self.new_encoder_layers:
            x = module(x)

        x = self.encode_to_latent(x)
        return x


class ThirdOrderCorrectionDecoder(nn.Module):
    def __init__(self, old_decoder_modules, new_channels_decoder,
                 last_out_channel=100, num_latent=7, num_node_channels=22,
                 nonlinearity=None, expand_type='individual', outer_type='individual',
                 pdim1=1):
        super().__init__()
        self.old_decoder_modules = old_decoder_modules
        self.decoder_polish = SnConv2(last_out_channel, new_channels_decoder[0], pdim1=pdim1)

        self.new_decoder_layers = nn.ModuleList()
        in_channel_i = calc_num_expand_channels(new_channels_decoder[0], expand_type)
        for out_channel_i in new_channels_decoder[1:]:
            self.new_decoder_layers.append(SnConv3(in_channel_i, out_channel_i))
            in_channel_i = out_channel_i
        self.final_conv = SnConv3to2(in_channel_i, 3,
                                     pdim1=pdim1)
        self.final_node_conv = SnConv3to1(in_channel_i, num_node_channels,
                                          pdim1=pdim1)
        if nonlinearity is None:
            nonlinearity = nn.ReLU
        self.nonlinearity = nonlinearity()

        self.expand_type = expand_type
        self.outer_type = outer_type
        self.pdim1 = pdim1

    def forward(self, x, mask=None):
        pdim1 = self.pdim1
        if self.outer_type == 'individual':
            x_t = x.unsqueeze(pdim1+1)
            x = x.unsqueeze(pdim1)
            x = x * x_t
        elif self.outer_type == 'all':
            x_t = x.unsqueeze(pdim1+1).unsqueeze(-1)
            x = x.unsqueeze(pdim1).unsqueeze(-2)
            x = x * x_t
            new_shape = x.shape[:-2] + (x.shape[-1] * x.shape[-2],)
            x = x.view(new_shape)

        for module in self.old_decoder_modules:
            x = module(x)
            x_node = None
            x = self.nonlinearity(x)
        x = self.decoder_polish(x)
        x = expand_2_to_3(x, mode=self.expand_type)

        for module in self.new_decoder_layers:
            x = module(x)
        x_out = self.final_conv(x)
        x_node = self.final_node_conv(x)
        return x_out, x_node
