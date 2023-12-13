import numpy as np
import torch
import torch.nn
import torch.nn.functional as F

from .sh import eval_sh_bases

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