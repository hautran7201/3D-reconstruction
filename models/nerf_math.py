# coding=utf-8
# Copyright 2022 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Mathy utility functions."""
import jax
import jax.numpy as jnp
import numpy as np
import torch


def get_freq_reg_mask(pos_enc_length, current_iter, total_reg_iter, max_visible=None, type='submission', device='cpu'):
  '''
  Returns a frequency mask for position encoding in NeRF.
  
  Args:
    pos_enc_length (int): Length of the position encoding.
    current_iter (int): Current iteration step.
    total_reg_iter (int): Total number of regularization iterations.
    max_visible (float, optional): Maximum visible range of the mask. Default is None. 
      For the demonstration study in the paper.
    
    Correspond to FreeNeRF paper:
      L: pos_enc_length
      t: current_iter
      T: total_iter
  
  Returns:
    jnp.array: Computed frequency or visibility mask.
  '''
  if max_visible is None:
    # default FreeNeRF
    if current_iter < total_reg_iter:
      freq_mask = torch.zeros(pos_enc_length).to(device)  # all invisible
      ptr = pos_enc_length / 3 * current_iter / total_reg_iter + 1 
      ptr = ptr if ptr < pos_enc_length / 3 else pos_enc_length / 3
      int_ptr = int(ptr)
      freq_mask[: int_ptr * 3] = 1.0  # assign the integer part
      freq_mask[int_ptr * 3 : int_ptr * 3 + 3] = (ptr - int_ptr)  # assign the fractional part
      return torch.clamp(freq_mask, 1e-8, 1 - 1e-8)
    else:
      return torch.ones(pos_enc_length).to(device)
  else:
    # For the ablation study that controls the maximum visible range of frequency spectrum
    freq_mask = torch.zeros(pos_enc_length).to(device)
    freq_mask[: int(pos_enc_length * max_visible)] = 1.0
    return freq_mask
  
  
def lossfun_occ_reg(rgb, density, reg_range=10, wb_prior=False, wb_range=20):
    '''
    Computes the occulusion regularization loss.

    Args:
        rgb (torch.Tensor): The RGB rays/images.
        density (torch.Tensor): The current density map estimate.
        reg_range (int): The number of initial intervals to include in the regularization mask.
        wb_prior (bool): If True, a prior based on the assumption of white or black backgrounds is used.
        wb_range (int): The range of RGB values considered to be a white or black background.

    Returns:
        float: The mean occlusion loss within the specified regularization range and white/black background region.
    '''
    # Compute the mean RGB value over the last dimension
    rgb_mean = rgb.mean(dim=-1)
    
    # Compute a mask for the white/black background region if using a prior
    if wb_prior:
        white_mask = (rgb_mean > 0.99).float()  # A naive way to locate white background
        black_mask = (rgb_mean < 0.01).float()  # A naive way to locate black background
        rgb_mask = (white_mask + black_mask)  # White or black background
        rgb_mask[:, wb_range:] = 0  # White or black background range
    else:
        rgb_mask = torch.zeros_like(rgb_mean)
    
    # Create a mask for the general regularization region
    if reg_range > 0:
        rgb_mask[:, :reg_range] = 1  # Penalize the points in reg_range close to the camera
    
    # Compute the density-weighted loss within the regularization and white/black background mask
    return (density * rgb_mask).mean()
## ------------------------------------------------------------------ ##


class EntropyLoss:
    def __init__(self, args, N_samples):
        super(EntropyLoss, self).__init__()
        self.N_samples = N_samples
        self.type_ = args.entropy_type 
        self.threshold = args.entropy_acc_threshold
        self.computing_entropy_all = args.computing_entropy_all
        self.smoothing = args.smoothing
        self.computing_ignore_smoothing = args.entropy_ignore_smoothing
        self.entropy_log_scaling = args.entropy_log_scaling
        self.N_entropy = args.N_entropy 
        
        if self.N_entropy ==0:
            self.computing_entropy_all = True
    
    def ray(self, density, acc):
        if self.smoothing and self.computing_ignore_smoothing:
            N_smooth = density.size(0)//2
            acc = acc[:N_smooth]
            density = density[:N_smooth]
        
        if not self.computing_entropy_all:
            acc = acc[self.N_samples:]
            density = density[self.N_samples:]
        
        density = torch.nn.functional.relu(density[...,-1])
        sigma = 1-torch.exp(-density)
        ray_prob = sigma / (torch.sum(sigma,-1).unsqueeze(-1)+1e-10)
        entropy_ray = torch.sum(self.entropy(ray_prob), -1)
        
        # masking no hitting poisition?
        mask = (acc>self.threshold).detach()
        entropy_ray*= mask
        entropy_ray_loss = torch.mean(entropy_ray, -1)
        if self.entropy_log_scaling:
            return torch.log(entropy_ray_loss + 1e-10)
        return entropy_ray_loss

    def ray_zvals(self, sigma, acc, N_samples):
        if self.smoothing and self.computing_ignore_smoothing:
            N_smooth = sigma.size(0)//2
            acc = acc[:N_smooth]
            sigma = sigma[:N_smooth]
        if not self.computing_entropy_all:
            acc = acc[N_samples:]
            sigma = sigma[N_samples:]

        """print(self.smoothing and self.computing_ignore_smoothing)
        print(not self.computing_entropy_all)
        print(sigma)
        print(acc)"""
        ray_prob = sigma / (torch.sum(sigma,-1).unsqueeze(-1)+1e-10)
        entropy_ray = self.entropy(ray_prob)
        """print(entropy_ray)"""
        entropy_ray_loss = torch.sum(entropy_ray, -1)
        
        # masking no hitting poisition?
        mask = (acc>self.threshold).detach()
        entropy_ray_loss*= mask
        if self.entropy_log_scaling:
            return torch.log(torch.mean(entropy_ray_loss) + 1e-10)
        return torch.mean(entropy_ray_loss)
    
    def ray_zvals_ver1_sigma(self, sigma, dists, acc):
        if self.smoothing and self.computing_ignore_smoothing:
            N_smooth = sigma.size(0)//2
            acc = acc[:N_smooth]
            sigma = sigma[:N_smooth]
            dists = dists[:N_smooth]
        
        if not self.computing_entropy_all:
            acc = acc[self.N_samples:]
            sigma = sigma[self.N_samples:]
            dists = dists[self.N_samples:]
        
        ray_prob = sigma / (torch.sum(sigma* dists,-1).unsqueeze(-1)+1e-10)
        entropy_ray = self.entropy(ray_prob)
        
        #intergral
        entropy_ray = entropy_ray * dists
        entropy_ray_loss = torch.sum(entropy_ray, -1)
        
        # masking no hitting poisition?
        mask = (acc>self.threshold).detach()
        entropy_ray_loss*= mask
        if self.entropy_log_scaling:
            return torch.log(torch.mean(entropy_ray_loss) + 1e-10)
        return torch.mean(entropy_ray_loss)

    def ray_zvals_ver2_alpha(self, alpha, dists, acc):
        if self.smoothing and self.computing_ignore_smoothing:
            N_smooth = alpha.size(0)//2
            acc = acc[:N_smooth]
            alpha = alpha[:N_smooth]
            dists = dists[:N_smooth]
        
        if not self.computing_entropy_all:
            acc = acc[self.N_samples:]
            alpha = alpha[self.N_samples:]
            dists = dists[self.N_samples:]
            
        ray_prob = alpha / (torch.sum(alpha,-1).unsqueeze(-1)+1e-10)
        
        entropy_ray = -1 * ray_prob * torch.log2(ray_prob/(dists+1e-10)+1e-10)
        entropy_ray_loss = torch.sum(entropy_ray, -1)
        # masking no hitting poisition?
        mask = (acc>self.threshold).detach()
        entropy_ray_loss*= mask
        if self.entropy_log_scaling:
            return torch.log(torch.mean(entropy_ray_loss) + 1e-10)
        return torch.mean(entropy_ray_loss)
    
    def entropy(self, prob):
        if self.type_ == 'log2':
            return -1*prob*torch.log2(prob+1e-10)
        elif self.type_ == '1-p':
            return prob*torch.log2(1-prob)


class SmoothingLoss:
    def __init__(self, args):
        super(SmoothingLoss, self).__init__()
    
        self.smoothing_activation = args.smoothing_activation
        self.criterion = torch.nn.KLDivLoss(reduction='batchmean')
    
    def __call__(self, sigma):
        half_num = sigma.size(0)//2
        sigma_1= sigma[:half_num]
        sigma_2 = sigma[half_num:]

        if self.smoothing_activation == 'softmax':
            p = F.softmax(sigma_1, -1)
            q = F.softmax(sigma_2, -1)
        elif self.smoothing_activation == 'norm':
            p = sigma_1 / (torch.sum(sigma_1, -1,  keepdim=True) + 1e-10) + 1e-10
            q = sigma_2 / (torch.sum(sigma_2, -1, keepdim=True) + 1e-10) +1e-10
        loss = self.criterion(p.log(), q)
        return loss