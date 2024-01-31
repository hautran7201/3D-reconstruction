import math 
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Optional, Union

from .sh import eval_sh_bases
from collections import OrderedDict


def positional_encoding(positions, freqs):
    freq_bands = (2**torch.arange(freqs).float()).to(positions.device)  # (F,)
    pts = (positions[..., None] * freq_bands).reshape(
        positions.shape[:-1] + (freqs * positions.shape[-1], ))  # (..., DF)
    pts = torch.cat([torch.sin(pts), torch.cos(pts)], dim=-1)
    return pts

def SHRender(xyz_sampled, viewdirs, features):
    sh_mult = eval_sh_bases(2, viewdirs)[:, None]
    rgb_sh = features.view(-1, 3, sh_mult.shape[-1])
    rgb = torch.relu(torch.sum(sh_mult * rgb_sh, dim=-1) + 0.5)
    return rgb


def RGBRender(xyz_sampled, viewdirs, features):

    rgb = features
    return rgb

class MLPRender_Fea(torch.nn.Module):
    def __init__(self,inChanel, viewpe=6, feape=6, featureC=128):
        super(MLPRender_Fea, self).__init__()

        self.in_mlpC = 2*viewpe*3 + 2*feape*inChanel + 3 + inChanel
        self.viewpe = viewpe
        self.feape = feape
        layer1 = torch.nn.Linear(self.in_mlpC, featureC)
        layer2 = torch.nn.Linear(featureC, featureC)
        layer3 = torch.nn.Linear(featureC,3)

        self.mlp = torch.nn.Sequential(layer1, torch.nn.ReLU(inplace=True), layer2, torch.nn.ReLU(inplace=True), layer3)
        torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, pts, viewdirs, features, mask):
        indata = [features, viewdirs]
        # print(indata[-1].shape)

        if self.feape > 0:
            encode = positional_encoding(features, self.feape)
            
            if mask['fea'] == None:    
                indata += [encode]
            else:
                indata += [encode*mask['fea']]

            # print(indata[-1].shape, encode.shape, mask['fea'].shape)

        if self.viewpe > 0:
            encode = positional_encoding(viewdirs, self.viewpe)

            if mask['view'] == None:    
                indata += [encode]
            else:
                indata += [encode*mask['view']]
            
            # print(indata[-1].shape, encode.shape, mask['view'].shape)

        mlp_in = torch.cat(indata, dim=-1)
        rgb = self.mlp(mlp_in)
        rgb = torch.sigmoid(rgb)

        return rgb

class MLPRender_PE(torch.nn.Module):
    def __init__(self,inChanel, viewpe=6, pospe=6, featureC=128):
        super(MLPRender_PE, self).__init__()

        self.in_mlpC = (3+2*viewpe*3)+ (2*pospe*3)  + inChanel #
        self.viewpe = viewpe
        self.pospe = pospe
        layer1 = torch.nn.Linear(self.in_mlpC, featureC)
        layer2 = torch.nn.Linear(featureC, featureC)
        layer3 = torch.nn.Linear(featureC,3)

        self.mlp = torch.nn.Sequential(layer1, torch.nn.ReLU(inplace=True), layer2, torch.nn.ReLU(inplace=True), layer3)
        torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, pts, viewdirs, features, mask):
        indata = [features, viewdirs]
        if self.pospe > 0:
            encode = positional_encoding(pts, self.pospe)

            if mask['pos'] == None:    
                indata += [encode]
            else:
                indata += [encode*mask['pos']]

        if self.viewpe > 0:
            encode = positional_encoding(viewdirs, self.viewpe)

            if mask['view'] == None:    
                indata += [encode]
            else:
                indata += [encode*mask['view']]
                
        mlp_in = torch.cat(indata, dim=-1)
        rgb = self.mlp(mlp_in)
        rgb = torch.sigmoid(rgb)

        return rgb

class MLPRender(torch.nn.Module):
    def __init__(self, inChanel, viewpe=6, pospe=6, feape=6, featureC=128):
        super(MLPRender, self).__init__()

        self.in_mlpC = (2*pospe*3) + (2*viewpe*3) + (2*feape*inChanel) + inChanel + 3
        self.viewpe = viewpe
        self.pospe = pospe
        self.feape = feape
        
        layer1 = torch.nn.Linear(self.in_mlpC, featureC)
        layer2 = torch.nn.Linear(featureC, featureC)
        layer3 = torch.nn.Linear(featureC,3)

        self.mlp = torch.nn.Sequential(layer1, torch.nn.ReLU(inplace=True), layer2, torch.nn.ReLU(inplace=True), layer3)
        torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, pts, viewdirs, features, mask):
        indata = [features, viewdirs]
        if self.pospe > 0:
            encode = positional_encoding(pts, self.pospe)

            if mask['pos'] == None:    
                indata += [encode]
            else:
                indata += [encode*mask['pos']]

        if self.viewpe > 0:
            encode = positional_encoding(viewdirs, self.viewpe)

            if mask['view'] == None:    
                indata += [encode]
            else:
                indata += [encode*mask['view']]

        if self.feape > 0:
            encode = positional_encoding(features, self.feape)
            
            if mask['fea'] == None:    
                indata += [encode]
            else:
                indata += [encode*mask['fea']]

        mlp_in = torch.cat(indata, dim=-1)
        rgb = self.mlp(mlp_in)
        rgb = torch.sigmoid(rgb)

        return rgb

class ComplexGaborLayer2D(nn.Module):
    '''
        Implicit representation with complex Gabor nonlinearity with 2D activation function
        
        Inputs;
            in_features: Input features
            out_features; Output features
            bias: if True, enable bias for the linear operation
            is_first: Legacy SIREN parameter
            omega_0: Legacy SIREN parameter
            omega0: Frequency of Gabor sinusoid term
            sigma0: Scaling of Gabor Gaussian term
            trainable: If True, omega and sigma are trainable parameters
    '''
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega0=10.0, sigma0=10.0,
                 trainable=False):
        super().__init__()
        self.omega_0 = omega0
        self.scale_0 = sigma0
        self.is_first = is_first
        
        self.in_features = in_features
        
        if self.is_first:
            dtype = torch.float
        else:
            dtype = torch.cfloat
            
        # Set trainable parameters if they are to be simultaneously optimized
        self.omega_0 = nn.Parameter(self.omega_0*torch.ones(1), trainable)
        self.scale_0 = nn.Parameter(self.scale_0*torch.ones(1), trainable)
        
        self.linear = nn.Linear(in_features,
                                out_features,
                                bias=bias,
                                dtype=dtype)
        
        # Second Gaussian window
        self.scale_orth = nn.Linear(in_features,
                                    out_features,
                                    bias=bias,
                                    dtype=dtype)
    
    def forward(self, input):
        lin = self.linear(input)
        
        scale_x = lin
        scale_y = self.scale_orth(input)
        
        freq_term = torch.exp(1j*self.omega_0*lin)
        
        arg = scale_x.abs().square() + scale_y.abs().square()
        gauss_term = torch.exp(-self.scale_0*self.scale_0*arg)
                
        return freq_term*gauss_term

class Wire(nn.Module):
    def __init__(
        self, 
        in_features, 
        hidden_features, 
        hidden_layers, 
        out_features, 
        pospe=6, 
        viewpe=6,        
        first_omega_0=10, 
        hidden_omega_0=10., 
        scale=10.0,
        sidelength=512, 
        fn_samples=None,
        use_nyquist=True
    ):

        super().__init__()
        
        # All results in the paper were with the default complex 'gabor' nonlinearity
        self.wire_layer = ComplexGaborLayer2D
        
        # Since complex numbers are two real numbers, reduce the number of 
        # hidden parameters by 4
        hidden_features = int(hidden_features/np.sqrt(2))
        dtype = torch.cfloat
        # self.wavelet = 'gabor'    

        # self.mlp = True
        # self.encoding = False
        self.viewpe = viewpe
        self.pospe = pospe
        # self.feape = 2
        inChanel = 27
        
        # Legacy parameter        
        self.net = []
        self.net.append(self.wire_layer(in_features,
                                    hidden_features, 
                                    omega0=first_omega_0,
                                    sigma0=scale,
                                    is_first=True,
                                    trainable=False))

        for i in range(hidden_layers):
            self.net.append(self.wire_layer(hidden_features,
                                        hidden_features, 
                                        omega0=hidden_omega_0,
                                        sigma0=scale))

        final_linear = nn.Linear(hidden_features,
                                 out_features,
                                 dtype=dtype)            
        self.net.append(final_linear)
        self.net = nn.Sequential(*self.net)

        """featureC = hidden_features*2
        layer1 = torch.nn.Linear(out_features+inChanel, featureC)
        layer2 = torch.nn.Linear(featureC, featureC)
        layer3 = torch.nn.Linear(featureC, featureC)
        layer4 = torch.nn.Linear(featureC,3)

        self.mlp = torch.nn.Sequential(
            layer1, 
            torch.nn.ReLU(inplace=True), 
            layer2, 
            torch.nn.ReLU(inplace=True), 
            layer4
        )

        torch.nn.init.constant_(self.mlp[-1].bias, 0)"""
    
    def forward(self, pts, viewdirs, features, mask):        
        data = torch.cat((pts, viewdirs, features), dim=1)
        output = self.net(data).real

        """if self.mlp:
            data = torch.cat((output, features), dim=1)
            output = self.mlp(data) """               

        output = torch.sigmoid(output)         

        return output

"""class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.
    
    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the 
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a 
    # hyperparameter.
    
    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of 
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))
    
    
class Siren(nn.Module):
    def __init__(
        self, 
        in_features, 
        hidden_features, 
        hidden_layers, 
        out_features, 
        outermost_linear=False, 
        first_omega_0=30, 
        hidden_omega_0=30.
    ):
        super().__init__()

        self.pos_encode = False
        
        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, 
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, 
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, 
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)

            self.net.append(final_linear)
            torch.nn.init.constant_(self.net[-1].bias, 0)

        else:
            self.net.append(SineLayer(hidden_features, out_features, 
                                      is_first=False, omega_0=hidden_omega_0))
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, pts, viewdirs, features, mask):

        if self.pos_encode:
            pts = positional_encoding(pts)
            viewdirs = positional_encoding(viewdirs)
            features = positional_encoding(features)

        data = torch.cat((pts, viewdirs, features), dim=1)
        # data = data.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        rgb = self.net(data)
        # rgb = torch.sigmoid(rgb)
        return rgb  """

class Sine(nn.Module):
    def __init__(self, w0=30.):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


class SirenLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, use_bias=True, w0=1., is_first=False):
        super().__init__()
        self.layer = nn.Linear(input_dim, hidden_dim, bias=use_bias)
        self.activation = Sine(w0)
        self.is_first = is_first
        self.input_dim = input_dim
        self.w0 = w0
        self.c = 6
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            dim = self.input_dim
            w_std = (1 / dim) if self.is_first else (math.sqrt(self.c / dim) / self.w0)
            self.layer.weight.uniform_(-w_std, w_std)
            if self.layer.bias is not None:
                self.layer.bias.uniform_(-w_std, w_std)

    def forward(self, x):
        out = self.layer(x) 
        out = self.activation(out)
        return out

class SirenNeRF(nn.Module):
    """
    SIREN-ized NeRF
    """
    def __init__(
        self,
        D: int = 8, skips: List = [4], W: int = 256,
        input_ch_appearance: int = 32,
        net_branch_appearance: bool = True,
        # siren related
        sigma_mul: float = 10.,
        rgb_mul: float = 1.,
        first_layer_w0: float = 30.,
        following_layers_w0: float = 1.,
        **not_used_kwargs
    ):
        super().__init__()
        self.skips = skips
        self.net_branch_appearance = net_branch_appearance
        self.sigma_mul = sigma_mul
        self.rgb_mul = rgb_mul

        input_ch_pts = 3    # fixed. do not change.
        use_bias = True     # fixed. do not change.
        # first_layer_w0 = 30.
        # following_layers_w0 = 1.

        base_layers = [SirenLayer(input_ch_pts, W, use_bias=use_bias, w0=first_layer_w0, is_first=True)]
        for _ in range(D-1):
            base_layers.append(SirenLayer(W, W, use_bias=use_bias, w0=following_layers_w0))
        dim = W
        self.base_layers = nn.Sequential(*base_layers)

        if self.net_branch_appearance:
            sigma_layers = [nn.Linear(dim, 1, bias=use_bias), ]
            self.sigma_layers = nn.Sequential(*sigma_layers)

            base_remap_layers = [nn.Linear(dim, W, bias=use_bias), ]
            self.base_remap_layers = nn.Sequential(*base_remap_layers)

            rgb_layers = [
                SirenLayer(W + input_ch_appearance, W // 2, use_bias=use_bias, w0=following_layers_w0),
            ] + [
                nn.Linear(W // 2, 3, bias=use_bias)]
            self.rgb_layers = nn.Sequential(*rgb_layers)
        else:
            output_layers = [nn.Linear(dim, 4, bias=use_bias), ]
            self.output_linear = nn.Sequential(*output_layers)

    def forward(
            self,
            input_pts: torch.Tensor,
            input_views: Optional[torch.Tensor] = None,
            input_features: Optional[torch.Tensor] = None,
            mask = None
        ):
        """
        input_pts:          [(B), N, 3]
        input_views:        [(B), N, any]
        """
        
        if input_views is None:
            shape = list(input_pts.shape)
            shape[-1] = 0
            input_views = input_pts.new_empty(shape)

        base = input_pts
        base = self.base_layers(input_pts)

        if self.net_branch_appearance:
            sigma: torch.Tensor = self.sigma_layers(base)
            base_remap = self.base_remap_layers(base)
            rgb = torch.cat((base_remap, input_views, input_features), dim=-1)
            rgb = self.rgb_layers(rgb)
        else:
            outputs = self.output_linear(base)
            rgb = outputs[..., :3]
            sigma = outputs[..., 3:]

        # only multiply on the positive side, since a large minus sigma is meaningless.
        # sigma = torch.where(sigma > 0, sigma * self.sigma_mul, sigma)
        
        # rgb values are normalized to [0, 1]
        rgb = torch.sigmoid_(rgb * self.rgb_mul)

        # ret = OrderedDict([('rgb', rgb), ('sigma', sigma.squeeze(-1))])
        return rgb     

class DoubleNeRF(nn.Module):
    def __init__(self,
                 base_cls: Union[NeRF, SirenNeRF],
                 net_kwargs: dict,
                 use_fine_model=False,
                 fine_kwargs: Optional[dict] = None,
                 **kwargs):
        super().__init__()

        self.use_fine_model = use_fine_model

        self.coarse_model = base_cls(**net_kwargs, **kwargs)
        if use_fine_model:
            assert fine_kwargs is not None
            self.fine_model = base_cls(**fine_kwargs, **kwargs)

    def forward(self, *args, is_coarse: bool = True, **kwargs):
        if is_coarse:
            return self.coarse_model(*args, **kwargs)
        else:
            assert self.use_fine_model
            return self.fine_model(*args, **kwargs)

    def query_sigma(self, *args, is_coarse: bool = True, **kwargs):
        if is_coarse:
            return self.coarse_model.query_sigma(*args, **kwargs,)
        else:
            assert self.use_fine_model
            return self.fine_model.query_sigma(*args, **kwargs,)
        