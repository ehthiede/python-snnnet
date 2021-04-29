import torch
import torch.nn as nn
from snnnet.second_order_layers import SnConv2, SnConv2to1, LocalConv2


class SnEncoder(nn.Module):
    """
    Second order encoder.  Encodes into a order-one vector

    Parameters
    ----------
    in_channels : int
        The size of the channel index of the input tensor
    hidden_channel_list : iterable of ints
        Number of channels in every hidden layer of the encoder
        Length corresponds to the number of layers.
    out_channels : int
        number of output channels.
    in_node_channels : int
        Number of input node channels.
    nonlinearity : None or pytorch module, optional:
        Nonlinearity to use.  If None, defaults to ReLU
    skip_conn : Skip connection, optional.
    """

    def __init__(self, in_channels, hidden_channel_list, out_channels, in_node_channels=0, nonlinearity=None, pdim1=1, skip_conn=False, diagonal_free=False, init_method=None):
        super().__init__()
        self.conv_layers = nn.ModuleList()
        in_channel_i = in_channels
        in_node_channel_i = in_node_channels
        for out_channel_i in hidden_channel_list:
            self.conv_layers.append(SnConv2(in_channel_i, out_channel_i, in_node_channels=in_node_channel_i, pdim1=pdim1, init_method=init_method, diagonal_free=diagonal_free))
            in_channel_i = out_channel_i
            in_node_channel_i = 0
        self.contractive_conv = SnConv2to1(in_channel_i, out_channels, in_node_channels=in_node_channel_i, pdim1=pdim1)
        if nonlinearity is None:
            nonlinearity = nn.ReLU
        self.nonlinearity = nonlinearity()
        self.pdim1 = pdim1

        self.skip_conn = skip_conn

    def forward(self, x, x_node=None, mask=None):
        """
        Runs a 2d convolution on the input data.

        Parameters
        ----------
        x : :class:`torch.Tensor`
            Input to the convolution, of shape B x n x n x C_in
            where B is the batch size, C_in is the number of channels,
            and n is the size of the permutation
        x_node : :class:`torch.Tensor` or None, optional
            Node features to input, of shape B x n x C_in
        mask : :class:`torch.Tensor`
            Mask used to account for graphs of unequal size.
            Index of the channel dimension, defaults to -1
        """
        if self.skip_conn:
            x_all = []

        for module in self.conv_layers:
            x = module(x, x_node=x_node, mask=mask)
            x_node = None
            x = self.nonlinearity(x)

            if self.skip_conn:
                x_all.append(x)

        if self.skip_conn:
            x = torch.cat(x_all, dim=3)

        x = self.contractive_conv(x, x_node=x_node)
        return x


class SnDecoder(nn.Module):
    """
    Second order encoder.  Encodes into a order-one vector

    Parameters
    ----------
    in_channels : int
        The size of the channel index of the input tensor
    hidden_channel_list : iterable of ints
        Number of channels in every hidden layer of the encoder
        Length corresponds to the number of layers.
    out_channels : int
        number of output channels.
    out_node_channels : int
        Number of node channels to output
    pdim1 : int
        First dimension on which the permutation group acts.
        Second dimension on which the group acts must be pdim1+1
    outer_type : string, optional
        How to handle the channels in the tensor product. Options are
        'individual' which maps to v_ic -> v_ic * v_jc, and "all" which maps
        v_ic -> v_ic * v_jd
    nonlinearity : None or pytorch module, optional:
        Nonlinearity to use.  If None, defaults to ReLU
    skip_conn : Skip connection, optional.
    """

    def __init__(self, in_channels, hidden_channel_list, out_channels, out_node_channels=0, pdim1=1, outer_type='individual', nonlinearity=None, skip_conn=False, diagonal_free=False, init_method=None, use_node_in=0, no_latent_node=False):
        super().__init__()
        self.outer_type = outer_type
        if self.outer_type == 'individual':
            in_channel_i = in_channels
        elif self.outer_type == 'all':
            in_channel_i = in_channels**2
        else:
            raise ValueError("Did not recognize outer type")

        self.conv_layers = nn.ModuleList()
        if no_latent_node:
            node_in = 0
        else:
            node_in = in_channels
        for out_channel_i in hidden_channel_list:
            self.conv_layers.append(SnConv2(in_channel_i, out_channel_i, in_node_channels=node_in, pdim1=pdim1, init_method=init_method, diagonal_free=diagonal_free))
            in_channel_i = out_channel_i
            node_in = 0
        self.final_conv = SnConv2(in_channel_i, out_channels, pdim1=pdim1)
        if out_node_channels > 0:
            self.final_node_conv = SnConv2to1(in_channel_i, out_node_channels, pdim1=pdim1)
        else:
            self.final_node_conv = None
        if nonlinearity is None:
            nonlinearity = nn.ReLU
        self.nonlinearity = nonlinearity()
        self.pdim1 = pdim1

        self.skip_conn = skip_conn

        self.no_latent_node = no_latent_node

    def forward(self, x, mask=None):
        """
        Runs a 2d convolution on the input data.

        Parameters
        ----------
        x : :class:`torch.Tensor`
            Input to the convolution, of shape B x n x n x C_in
            where B is the batch size, C_in is the number of channels,
            and n is the size of the permutation
        mask : :class:`torch.Tensor`
            Mask used to account for graphs of unequal size.
            Index of the channel dimension, defaults to -1
        pdim1 : int
            First dimension on which the permutation group acts.
            Second dimension on which the group acts must be pdim1+1
        """
        pdim1 = self.pdim1
        if self.outer_type == 'individual':
            x_t = x.unsqueeze(pdim1+1)
            x_2d = x.unsqueeze(pdim1)
            x_2d = x_2d * x_t
        elif self.outer_type == 'all':
            x_t = x.unsqueeze(pdim1+1).unsqueeze(-1)
            x_2d = x.unsqueeze(pdim1).unsqueeze(-2)
            x_2d = x_2d * x_t
            new_shape = x_2d.shape[:-2] + (x_2d.shape[-1] * x_2d.shape[-2],)
            x_2d = x_2d.view(new_shape)
        if mask is not None:
            x_2d[~mask] = 0.

        if self.skip_conn:
            x_all = []
        if self.no_latent_node:
            x_node = None
        else:
            x_node = x
        x = x_2d
        for module in self.conv_layers:
            x = module(x, mask=mask, x_node=x_node)
            x_node = None
            x = self.nonlinearity(x)

            if self.skip_conn:
                x_all.append(x)

        if self.skip_conn:
            x = torch.cat(x_all, dim=3)

        x_out = self.final_conv(x, mask=mask)
        if self.final_node_conv is not None:
            x_node = self.final_node_conv(x, mask=mask)
            return x_out, x_node
        else:
            return x_out


class SnVAE(nn.Module):
    """
    Permutation Variational Autoencoder
    """

    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, x_node=None, mask=None):
        # Encoder
        encoded_vars = self.encoder(x, x_node, mask)
        N_channels = encoded_vars.shape[-1]
        assert(N_channels % 2 == 0), "Number of channels is not even!"
        z_var = encoded_vars[..., :N_channels//2]
        z_mu = encoded_vars[..., N_channels//2:]

        # Re-parameterization trick: Sample from the distribution having latent parameters z_mu, z_var
        std = torch.exp(z_var / 2)
        eps = torch.randn_like(std)
        x_sample = eps.mul(std).add_(z_mu)

        # Sigmoid on the output of the decoder
        '''
        if x_node is not None:
            # Decoder
            predict_x, predict_x_node = self.decoder(x_sample, mask)

            # Sigmoid
            predict_x = torch.sigmoid(predict_x)

            return (predict_x, predict_x_node), z_mu, z_var

        # Decoder
        predict = self.decoder(x_sample, mask)

        # Sigmoid
        predict = torch.sigmoid(predict)
        '''

        # Decoder
        predict = self.decoder(x_sample, mask)

        return predict, z_mu, z_var


class SnAE(nn.Module):
    """
    Permutation Autoencoder
    """

    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, x_node=None, mask=None):
        # Encoder
        encoded = self.encoder(x, x_node, mask)

        # Decoder
        predict = self.decoder(encoded, mask)

        return predict


class SnPredictor(nn.Module):
    """
    Permutation Predictor
    """

    def __init__(self, encoder, in_channels, out_channels, nonlinearity=None):
        super().__init__()

        self.encoder = encoder
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nonlinearity = nonlinearity
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_channels, out_channels))

    def forward(self, x, x_node=None, mask=None):
        # Encoder
        encoded = self.encoder(x, x_node, mask)

        # Reshape encoded
        predict = torch.reshape(encoded, (encoded.size(0), encoded.nelement() // encoded.size(0)))

        assert predict.size(1) == self.in_channels

        # Prediction
        for module in self.layers:
            predict = module(predict)
            if self.nonlinearity is not None:
                predict = self.nonlinearity(predict)
        return predict


class SnNodeChannels(nn.Module):
    """
    Node channels compression
    """

    def __init__(self, in_channels, out_channels, hidden_channel_list=None, nonlinearity=None):
        super().__init__()
        if nonlinearity is None:
            nonlinearity = nn.Sigmoid()
        self.nonlinearity = nonlinearity
        self.layers = nn.ModuleList()
        if hidden_channel_list is None:
            self.layers.append(nn.Linear(in_channels, out_channels))
        else:
            nLayers = len(hidden_channel_list)
            for layer in range(nLayers):
                if layer == 0:
                    self.layers.append(nn.Linear(in_channels, hidden_channel_list[layer]))
                else:
                    self.layers.append(nn.Linear(hidden_channel_list[layer - 1], hidden_channel_list[layer]))
            self.layers.append(nn.Linear(hidden_channel_list[nLayers - 1], out_channels))

    def forward(self, x_node):
        """
        Runs a node channel compression using MLP.

        Parameters
        ----------
        x_node : :class:`torch.Tensor`
            Node features to input, of shape B x n x C_in
        """
        for module in self.layers:
            x_node = module(x_node)
            x_node = self.nonlinearity(x_node)
        return x_node


class SnLocalEncoder(nn.Module):
    """
    Second order encoder.  Encodes into a order-one vector

    Parameters
    ----------
    in_channels : int
        The size of the channel index of the input tensor
    hidden_channel_list : iterable of ints
        Number of channels in every hidden layer of the encoder
        Length corresponds to the number of layers.
    out_channels : int
        number of output channels.
    in_node_channels : int
        Number of input node channels.
    nonlinearity : None or pytorch module, optional:
        Nonlinearity to use.  If None, defaults to ReLU
    """

    def __init__(self, in_channels, hidden_channel_list, neighborhood_channels, out_channels, in_node_channels=0, pdim1=1, nonlinearity=None, product_type='elementwise'):
        super().__init__()
        self.local_conv_layers = nn.ModuleList()
        self.basic_conv_layers = nn.ModuleList()
        in_channel_i = in_channels
        in_node_channel_i = in_node_channels
        for n_channel_i, out_channel_i in zip(neighborhood_channels, hidden_channel_list):
            self.basic_conv_layers.append(SnConv2(in_channel_i, in_channel_i + in_node_channel_i, in_node_channels=in_node_channel_i, pdim1=pdim1))
            self.local_conv_layers.append(LocalConv2(in_channel_i + in_node_channel_i, n_channel_i, out_channel_i, in_node_channels=0, pdim1=pdim1, product_type=product_type))
            in_channel_i = out_channel_i
            in_node_channel_i = 0
        self.contractive_conv = SnConv2to1(in_channel_i, out_channels, in_node_channels=in_node_channel_i, pdim1=pdim1)
        if nonlinearity is None:
            nonlinearity = nn.ReLU
        self.nonlinearity = nonlinearity()
        self.pdim1 = pdim1

    def forward(self, x, x_node=None, mask=None):
        """
        Runs a 2d convolution on the input data.

        Parameters
        ----------
        x : :class:`torch.Tensor`
            Input to the convolution, of shape B x n x n x C_in
            where B is the batch size, C_in is the number of channels,
            and n is the size of the permutation
        x_node : :class:`torch.Tensor` or None, optional
            Node features to input, of shape B x n x C_in
        mask : :class:`torch.Tensor`
            Mask used to account for graphs of unequal size.
            Index of the channel dimension, defaults to -1
        """
        for basic_module, local_module in zip(self.basic_conv_layers, self.local_conv_layers):
            x = basic_module(x, x_node=x_node, mask=mask)
            x_node = None
            x = local_module(x, x_node=x_node, mask=mask)
            x = self.nonlinearity(x)
            if mask is not None:
                x[~mask] = 0.
        x = self.contractive_conv(x, x_node=x_node)
        x = torch.mean(x, dim=self.pdim1+1)
        return x


class SnLocalDecoder(nn.Module):
    """
    Second order encoder.  Encodes into a order-one vector

    Parameters
    ----------
    in_channels : int
        The size of the channel index of the input tensor
    hidden_channel_list : iterable of ints
        Number of channels in every hidden layer of the encoder
        Length corresponds to the number of layers.
    out_channels : int
        number of output channels.
    out_node_channels : int
        Number of node channels to output
    pdim1 : int
        First dimension on which the permutation group acts.
        Second dimension on which the group acts must be pdim1+1
    outer_type : string, optional
        How to handle the channels in the tensor product. Options are
        'individual' which maps to v_ic -> v_ic * v_jc, and "all" which maps
        v_ic -> v_ic * v_jd
    nonlinearity : None or pytorch module, optional:
        Nonlinearity to use.  If None, defaults to ReLU
    """

    def __init__(self, in_channels, hidden_channel_list, neighborhood_channels, out_channels, out_node_channels=0, pdim1=1, outer_type='individual', nonlinearity=None, product_type='elementwise'):
        super().__init__()
        self.outer_type = outer_type
        if self.outer_type == 'individual':
            in_channel_i = in_channels
        elif self.outer_type == 'all':
            in_channel_i = in_channels**3
        else:
            raise ValueError("Did not recognize outer type")

        self.local_conv_layers = nn.ModuleList()
        self.basic_conv_layers = nn.ModuleList()
        for n_channel_i, out_channel_i in zip(neighborhood_channels, hidden_channel_list):
            self.basic_conv_layers.append(SnConv2(in_channel_i, in_channel_i, pdim1=pdim1))
            self.local_conv_layers.append(LocalConv2(in_channel_i, n_channel_i, out_channel_i, in_node_channels=0, pdim1=pdim1, product_type=product_type))
            in_channel_i = out_channel_i
        self.final_conv = SnConv2(in_channel_i, out_channels, pdim1=pdim1)
        if out_node_channels > 0:
            self.final_node_conv = SnConv2to1(in_channel_i, out_node_channels, pdim1=pdim1)
        else:
            self.final_node_conv = None
        if nonlinearity is None:
            nonlinearity = nn.ReLU
        self.nonlinearity = nonlinearity()
        self.pdim1 = pdim1

    def forward(self, x, mask=None):
        """
        Runs a 2d convolution on the input data.

        Parameters
        ----------
        x : :class:`torch.Tensor`
            Input to the convolution, of shape B x n x n x C_in
            where B is the batch size, C_in is the number of channels,
            and n is the size of the permutation
        mask : :class:`torch.Tensor`
            Mask used to account for graphs of unequal size.
            Index of the channel dimension, defaults to -1
        pdim1 : int
            First dimension on which the permutation group acts.
            Second dimension on which the group acts must be pdim1+1
        """
        pdim1 = self.pdim1
        if self.outer_type == 'individual':
            new_x = x.unsqueeze(pdim1) * x.unsqueeze(pdim1+1)
            x_tt = x.unsqueeze(pdim1).unsqueeze(pdim1+1)
            x = new_x.unsqueeze(pdim1+2) * x_tt
        elif self.outer_type == 'all':
            # x_t = x.unsqueeze(pdim1+1).unsqueeze(-1)
            # x = x.unsqueeze(pdim1).unsqueeze(-2)
            # x = x * x_t
            # new_shape = x.shape[:-2] + (x.shape[-1] * x.shape[-2],)
            # x = x.view(new_shape)
            raise RuntimeError("Not implememented yet!")
        if mask is not None:
            x[~mask] = 0.

        for basic_module, local_module in zip(self.basic_conv_layers, self.local_conv_layers):
            x = basic_module(x, mask=mask)
            x = local_module(x, mask=mask)
            x = self.nonlinearity(x)
            if mask is not None:
                x[~mask] = 0.
        x_out = self.final_conv(x, mask=mask)
        if self.final_node_conv is not None:
            x_node = self.final_node_conv(x, mask=mask)
            return x_out, x_node
        else:
            return x_out
