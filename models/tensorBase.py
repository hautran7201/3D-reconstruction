import time

import numpy as np
import torch
import torch.nn
import torch.nn.functional as F

from .sh import eval_sh_bases
from .mlp import *


def positional_encoding(positions, freqs):
    
        freq_bands = (2**torch.arange(freqs).float()).to(positions.device)  # (F,)
        pts = (positions[..., None] * freq_bands).reshape(
            positions.shape[:-1] + (freqs * positions.shape[-1], ))  # (..., DF)
        pts = torch.cat([torch.sin(pts), torch.cos(pts)], dim=-1)
        return pts

def raw2alpha(sigma, dist):
    # sigma, dist  [N_rays, N_samples]
    alpha = 1. - torch.exp(-sigma*dist)

    T = torch.cumprod(torch.cat([torch.ones(alpha.shape[0], 1).to(alpha.device), 1. - alpha + 1e-10], -1), -1)

    weights = alpha * T[:, :-1]  # [N_rays, N_samples]
    return alpha, weights, T[:,-1:]

class AlphaGridMask(torch.nn.Module):
    def __init__(self, device, aabb, alpha_volume):
        super(AlphaGridMask, self).__init__()
        self.device = device

        self.aabb=aabb.to(self.device)
        self.aabbSize = self.aabb[1] - self.aabb[0]
        self.invgridSize = 1.0/self.aabbSize * 2
        self.alpha_volume = alpha_volume.view(1,1,*alpha_volume.shape[-3:])
        self.gridSize = torch.LongTensor([alpha_volume.shape[-1],alpha_volume.shape[-2],alpha_volume.shape[-3]]).to(self.device)

    def sample_alpha(self, xyz_sampled):
        xyz_sampled = self.normalize_coord(xyz_sampled)
        alpha_vals = F.grid_sample(self.alpha_volume, xyz_sampled.view(1,-1,1,1,3), align_corners=True).view(-1)

        return alpha_vals

    def normalize_coord(self, xyz_sampled):
        return (xyz_sampled-self.aabb[0]) * self.invgridSize - 1


class TensorBase(torch.nn.Module):
    def __init__(self, args, aabb, gridSize, device, density_n_comp = 8, appearance_n_comp = 24, app_dim = 27,
                    shadingMode = 'MLP_PE', alphaMask = None, occGrid = None, near_far=[2.0,6.0],
                    density_shift = -10, alphaMask_thres=0.001, distance_scale=25, rayMarch_weight_thres=0.0001,
                    pos_pe = 6, view_pe = 6, fea_pe = 6, featureC=128, step_ratio=2.0,
                    fea2denseAct = 'softplus'):
        super(TensorBase, self).__init__()

        self.density_n_comp = density_n_comp
        self.app_n_comp = appearance_n_comp
        self.app_dim = app_dim
        self.aabb = aabb
        self.alphaMask = alphaMask
        self.occGrid = occGrid
        self.device=device

        self.density_shift = density_shift
        self.alphaMask_thres = alphaMask_thres
        self.distance_scale = distance_scale
        self.rayMarch_weight_thres = rayMarch_weight_thres
        self.fea2denseAct = fea2denseAct

        self.near_far = near_far
        self.step_ratio = step_ratio


        self.update_stepSize(gridSize)

        self.matMode = [[0,1], [0,2], [1,2]]
        self.vecMode =  [2, 1, 0]
        self.comp_w = [1,1,1]


        self.init_svd_volume(gridSize[0], device)

        self.pos_bit_length = [2*pos_pe*3]
        self.view_bit_length = [2*view_pe*3]
        self.fea_bit_length = [2*fea_pe*app_dim]

        self.density_bit_length = density_n_comp
        self.app_bit_length = appearance_n_comp

        self.shadingMode, self.pos_pe, self.view_pe, self.fea_pe, self.featureC = shadingMode, pos_pe, view_pe, fea_pe, featureC
        self.init_render_func(shadingMode, pos_pe, view_pe, fea_pe, featureC, device)

    def init_render_func(self, shadingMode, pos_pe, view_pe, fea_pe, featureC, device):
        if shadingMode == 'MLP_PE':
            self.renderModule = MLPRender_PE(
                self.app_dim, 
                view_pe, 
                pos_pe, 
                featureC
            ).to(device)
        elif shadingMode == 'MLP_Fea':
            self.renderModule = MLPRender_Fea(
                self.app_dim, 
                view_pe, 
                fea_pe, 
                featureC
            ).to(device)
        elif shadingMode == 'MLP':
            self.renderModule = MLPRender(
                self.app_dim, 
                view_pe, 
                pos_pe, 
                fea_pe, 
                featureC
            ).to(device)
        elif shadingMode == 'Siren':
            self.renderModule = SirenNeRF(
                D = 4, 
                skips = [2], 
                W = 256,
                input_ch_appearance = 3+27,
                net_branch_appearance = True,

                # siren related
                sigma_mul = 10.,
                rgb_mul = 1.,
                first_layer_w0 = 30.,
                following_layers_w0 = 1.,
            ).to(device)
        elif shadingMode == 'Wire':
            self.renderModule = Wire(
                in_features=3+3+self.app_dim, 
                hidden_features=128, 
                hidden_layers=3, 
                out_features=3,
                first_omega_0=40,
                hidden_omega_0=40,
                scale=40

            ).to(device)
        else:
            print("Unrecognized shading module")
            exit()

        """elif shadingMode == 'Siren':
            self.renderModule = Siren(
                in_features=3+3+self.app_dim, 
                hidden_features=featureC, 
                hidden_layers=4, 
                out_features=3,
                outermost_linear=True
            ).to(device)"""

        print("pos_pe", pos_pe, "view_pe", view_pe, "fea_pe", fea_pe)
        print(self.renderModule)

    def update_stepSize(self, gridSize):
        print("aabb", self.aabb.view(-1))
        print("grid size", gridSize)
        self.aabbSize = self.aabb[1] - self.aabb[0]
        self.invaabbSize = 2.0/self.aabbSize
        self.gridSize= torch.LongTensor(gridSize).to(self.device)
        self.units=self.aabbSize / (self.gridSize-1)
        self.stepSize=torch.mean(self.units)*self.step_ratio
        self.aabbDiag = torch.sqrt(torch.sum(torch.square(self.aabbSize)))
        self.nSamples=int((self.aabbDiag / self.stepSize).item()) + 1
        print("sampling step size: ", self.stepSize)
        print("sampling number: ", self.nSamples)

    def init_svd_volume(self, res, device):
        pass

    def compute_features(self, xyz_sampled):
        pass
    
    def compute_densityfeature(self, xyz_sampled):
        pass
    
    def compute_appfeature(self, xyz_sampled):
        pass
    
    def normalize_coord(self, xyz_sampled):
        return (xyz_sampled-self.aabb[0]) * self.invaabbSize - 1

    def get_optparam_groups(self, lr_init_spatial = 0.02, lr_init_network = 0.001):
        pass

    def get_kwargs(self):
        return {
            'aabb': self.aabb,
            'gridSize':self.gridSize.tolist(),
            'occGrid':self.occGrid,
            'density_n_comp': self.density_n_comp,
            'appearance_n_comp': self.app_n_comp,
            'app_dim': self.app_dim,

            'density_shift': self.density_shift,
            'alphaMask_thres': self.alphaMask_thres,
            'distance_scale': self.distance_scale,
            'rayMarch_weight_thres': self.rayMarch_weight_thres,
            'fea2denseAct': self.fea2denseAct,

            'near_far': self.near_far,
            'step_ratio': self.step_ratio,

            'shadingMode': self.shadingMode,
            'pos_pe': self.pos_pe,
            'view_pe': self.view_pe,
            'fea_pe': self.fea_pe,
            'featureC': self.featureC
        }

    def save(self, path):
        kwargs = self.get_kwargs()
        ckpt = {'kwargs': kwargs, 'state_dict': self.state_dict()}
        if self.alphaMask is not None:
            alpha_volume = self.alphaMask.alpha_volume.bool().cpu().numpy()
            ckpt.update({'alphaMask.shape':alpha_volume.shape})
            ckpt.update({'alphaMask.mask':np.packbits(alpha_volume.reshape(-1))})
            ckpt.update({'alphaMask.aabb': self.alphaMask.aabb.cpu()})
        torch.save(ckpt, path)

    def load(self, ckpt):
        if 'alphaMask.aabb' in ckpt.keys():
            length = np.prod(ckpt['alphaMask.shape'])
            alpha_volume = torch.from_numpy(np.unpackbits(ckpt['alphaMask.mask'])[:length].reshape(ckpt['alphaMask.shape']))
            self.alphaMask = AlphaGridMask(self.device, ckpt['alphaMask.aabb'].to(self.device), alpha_volume.float().to(self.device))
        self.load_state_dict(ckpt['state_dict'])


    def sample_ray_ndc(self, rays_o, rays_d, is_train=True, N_samples=-1):
        N_samples = N_samples if N_samples > 0 else self.nSamples
        near, far = self.near_far
        interpx = torch.linspace(near, far, N_samples).unsqueeze(0).to(rays_o)
        if is_train:
            interpx += torch.rand_like(interpx).to(rays_o) * ((far - near) / N_samples)

        rays_pts = rays_o[..., None, :] + rays_d[..., None, :] * interpx[..., None]
        mask_outbbox = ((self.aabb[0] > rays_pts) | (rays_pts > self.aabb[1])).any(dim=-1)
        return rays_pts, interpx, ~mask_outbbox

    def sample_ray(self, rays_o, rays_d, is_train=True, N_samples=-1):
        N_samples = N_samples if N_samples>0 else self.nSamples
        stepsize = self.stepSize
        near, far = self.near_far
        vec = torch.where(rays_d==0, torch.full_like(rays_d, 1e-6), rays_d)
        rate_a = (self.aabb[1] - rays_o) / vec
        rate_b = (self.aabb[0] - rays_o) / vec
        t_min = torch.minimum(rate_a, rate_b).amax(-1).clamp(min=near, max=far)

        rng = torch.arange(N_samples)[None].float()
        if is_train:
            rng = rng.repeat(rays_d.shape[-2],1)
            rng += torch.rand_like(rng[:,[0]])
        step = stepsize * rng.to(rays_o.device)
        interpx = (t_min[...,None] + step)

        rays_pts = rays_o[...,None,:] + rays_d[...,None,:] * interpx[...,None]
        mask_outbbox = ((self.aabb[0]>rays_pts) | (rays_pts>self.aabb[1])).any(dim=-1)

        return rays_pts, interpx, ~mask_outbbox


    def shrink(self, new_aabb, voxel_size):
        pass

    @torch.no_grad()
    def getDenseAlpha(self,gridSize=None):
        gridSize = self.gridSize if gridSize is None else gridSize

        samples = torch.stack(torch.meshgrid(
            torch.linspace(0, 1, gridSize[0]),
            torch.linspace(0, 1, gridSize[1]),
            torch.linspace(0, 1, gridSize[2]),
        ), -1).to(self.device)
        dense_xyz = self.aabb[0] * (1-samples) + self.aabb[1] * samples

        # dense_xyz = dense_xyz
        # print(self.stepSize, self.distance_scale*self.aabbDiag)
        alpha = torch.zeros_like(dense_xyz[...,0])
        for i in range(gridSize[0]):
            alpha[i] = self.compute_alpha(dense_xyz[i].view(-1,3), self.stepSize).view((gridSize[1], gridSize[2]))
        return alpha, dense_xyz

    @torch.no_grad()
    def updateAlphaMask(self, gridSize=(200,200,200)):

        alpha, dense_xyz = self.getDenseAlpha(gridSize)
        dense_xyz = dense_xyz.transpose(0,2).contiguous()
        alpha = alpha.clamp(0,1).transpose(0,2).contiguous()[None,None]
        total_voxels = gridSize[0] * gridSize[1] * gridSize[2]

        ks = 3
        alpha = F.max_pool3d(alpha, kernel_size=ks, padding=ks // 2, stride=1).view(gridSize[::-1])
        alpha[alpha>=self.alphaMask_thres] = 1
        alpha[alpha<self.alphaMask_thres] = 0

        self.alphaMask = AlphaGridMask(self.device, self.aabb, alpha)

        valid_xyz = dense_xyz[alpha>0.5]

        xyz_min = valid_xyz.amin(0)
        xyz_max = valid_xyz.amax(0)

        new_aabb = torch.stack((xyz_min, xyz_max))

        total = torch.sum(alpha)
        print(f"bbox: {xyz_min, xyz_max} alpha rest %%%f"%(total/total_voxels*100))
        return new_aabb

    @torch.no_grad()
    def filtering_rays(self, all_rays, all_rgbs, N_samples=256, chunk=10240*5, bbox_only=False):
        print('========> filtering rays ...')
        tt = time.time()

        N = torch.tensor(all_rays.shape[:-1]).prod()

        mask_filtered = []
        idx_chunks = torch.split(torch.arange(N), chunk)
        for idx_chunk in idx_chunks:
            rays_chunk = all_rays[idx_chunk].to(self.device)

            rays_o, rays_d = rays_chunk[..., :3], rays_chunk[..., 3:6]
            if bbox_only:
                vec = torch.where(rays_d == 0, torch.full_like(rays_d, 1e-6), rays_d)
                rate_a = (self.aabb[1] - rays_o) / vec
                rate_b = (self.aabb[0] - rays_o) / vec
                t_min = torch.minimum(rate_a, rate_b).amax(-1)#.clamp(min=near, max=far)
                t_max = torch.maximum(rate_a, rate_b).amin(-1)#.clamp(min=near, max=far)
                mask_inbbox = t_max > t_min

            else:
                xyz_sampled, _,_ = self.sample_ray(rays_o, rays_d, N_samples=N_samples, is_train=False)
                mask_inbbox= (self.alphaMask.sample_alpha(xyz_sampled).view(xyz_sampled.shape[:-1]) > 0).any(-1)

            mask_filtered.append(mask_inbbox.cpu())

        mask_filtered = torch.cat(mask_filtered).view(all_rgbs.shape[:-1])

        print(f'Ray filtering done! takes {time.time()-tt} s. ray mask ratio: {torch.sum(mask_filtered) / N}')
        return all_rays[mask_filtered], all_rgbs[mask_filtered]


    def feature2density(self, density_features):
        if self.fea2denseAct == "softplus":
            return F.softplus(density_features+self.density_shift)
        elif self.fea2denseAct == "relu":
            return F.relu(density_features)


    def compute_alpha(self, xyz_locs, length=1):

        if self.alphaMask is not None:
            alphas = self.alphaMask.sample_alpha(xyz_locs)
            alpha_mask = alphas > 0
        else:
            alpha_mask = torch.ones_like(xyz_locs[:,0], dtype=bool)
            

        sigma = torch.zeros(xyz_locs.shape[:-1], device=xyz_locs.device)

        if alpha_mask.any():
            xyz_sampled = self.normalize_coord(xyz_locs[alpha_mask])
            sigma_feature = self.compute_densityfeature(xyz_sampled, None)
            validsigma = self.feature2density(sigma_feature)
            sigma[alpha_mask] = validsigma
        

        alpha = 1 - torch.exp(-sigma*length).view(xyz_locs.shape[:-1])

        return alpha


    def _forward(self, rays_chunk, mask, white_bg=True, is_train=False, ndc_ray=False, N_samples=-1):
        if mask == None:
            encoding_mask = {
                'pos': None,
                'view': None,
                'fea': None
            }
            den_decomp_mask = None
            app_decomp_mask = None
        else:
            encoding_mask = mask['encoding']
            den_decomp_mask = mask['decomp']['den']
            app_decomp_mask = mask['decomp']['app']

        # sample points
        viewdirs = rays_chunk[:, 3:6]
        if ndc_ray:
            xyz_sampled, z_vals, ray_valid = self.sample_ray_ndc(rays_chunk[:, :3], viewdirs, is_train=is_train,N_samples=N_samples)
            dists = torch.cat((z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])), dim=-1)
            rays_norm = torch.norm(viewdirs, dim=-1, keepdim=True)
            dists = dists * rays_norm
            viewdirs = viewdirs / rays_norm
        else:
            xyz_sampled, z_vals, ray_valid = self.sample_ray(rays_chunk[:, :3], viewdirs, is_train=is_train,N_samples=N_samples)
            dists = torch.cat((z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])), dim=-1)
        viewdirs = viewdirs.view(-1, 1, 3).expand(xyz_sampled.shape)
        
        if self.alphaMask is not None:
            alphas = self.alphaMask.sample_alpha(xyz_sampled[ray_valid])
            alpha_mask = alphas > 0
            ray_invalid = ~ray_valid
            ray_invalid[ray_valid] |= (~alpha_mask)
            ray_valid = ~ray_invalid


        sigma = torch.zeros(xyz_sampled.shape[:-1], device=xyz_sampled.device)
        rgb = torch.zeros((*xyz_sampled.shape[:2], 3), device=xyz_sampled.device)

        if ray_valid.any():
            xyz_sampled = self.normalize_coord(xyz_sampled)
            sigma_feature = self.compute_densityfeature(xyz_sampled[ray_valid], mask=den_decomp_mask)

            validsigma = self.feature2density(sigma_feature)
            sigma[ray_valid] = validsigma


        alpha, weight, bg_weight = raw2alpha(sigma, dists * self.distance_scale)

        app_mask = weight > self.rayMarch_weight_thres

        if app_mask.any():
            app_features = self.compute_appfeature(xyz_sampled[app_mask], mask=app_decomp_mask)
            valid_rgbs = self.renderModule(xyz_sampled[app_mask], viewdirs[app_mask], app_features, mask=encoding_mask)
            rgb[app_mask] = valid_rgbs

        acc_map = torch.sum(weight, -1)
        rgb_map = torch.sum(weight[..., None] * rgb, -2)
        rgb_map_1 = rgb_map

        if white_bg: #  or (is_train and torch.rand((1,))<0.5)
            rgb_map = rgb_map + (1. - acc_map[..., None])

        rgb_map = rgb_map.clamp(0,1)

        with torch.no_grad():
            depth_map = torch.sum(weight * z_vals, -1) #  / acc_map
            depth_map = depth_map + (1. - acc_map)
            if torch.isnan(rgb_map).any().item():
                print()
                print(torch.isnan(rgb_map).any().item())
                print(torch.isnan(rgb_map_1).any().item())
                print(torch.isnan(weight[..., None]).any().item())
                print(torch.isnan(rgb).any().item())
                print(torch.min(rgb_map))
                print(torch.max(rgb_map))
                print('have NAN')
                print()
                
                if ray_valid.any():
                    print('If ray any')
                    print(torch.isnan(xyz_sampled).any().item())
                    print(torch.isnan(sigma_feature).any().item())                    
                    print(torch.isnan(validsigma).any().item()) 
                    print(sigma_feature)
                exit()
            disparity_map = depth_map + (1. - acc_map) * rays_chunk[..., -1]

        num_valid_samples = app_mask.long().sum()

        # dist loss
        sample_dist = 1. / N_samples
        z_vals_shifted = torch.cat(
            [z_vals[..., 1:], sample_dist * torch.ones_like(z_vals[..., :1])], dim=-1
        )

        mid_zs = 0.5 * z_vals + 0.5 * z_vals_shifted  # [N, T]
        loss_dist_per_ray = (torch.abs(mid_zs.unsqueeze(1) - mid_zs.unsqueeze(2)) *
                                (weight.unsqueeze(1) * weight.unsqueeze(2))).sum(dim=[1, 2]) \
                    + 1/3 * ((z_vals_shifted - z_vals) * (weight ** 2)).sum(dim=1)  # [N]

        loss_dist_per_ray = loss_dist_per_ray / (torch.sum(weight * z_vals, dim=-1) + 1e-6)                    
        
        loss_dist_cutoff = 1e-3
        loss_dist_per_ray[loss_dist_per_ray < loss_dist_cutoff] = 0.
        loss_dist = loss_dist_per_ray.sum()

        outputs = {
            'rgb_map': rgb_map,
            'disparity_map': disparity_map,
            'depth_map': depth_map,
            'num_valid_samples': num_valid_samples,
            'loss_dist': loss_dist.view(1,),
            'weight': weight,
            'rgb': rgb
        }

        return outputs

    def _forward_nerfacc(
        self,
        rays_chunk,
        mask,
        white_bg=True,
        is_train=False,
        ndc_ray=False,
        N_samples=-1,
    ):

        if mask == None:
            encoding_mask = {
                'pos': None,
                'view': None,
                'fea': None
            }
            den_decomp_mask = None
            app_decomp_mask = None
        else:
            encoding_mask = mask['encoding']
            den_decomp_mask = mask['decomp']['den']
            app_decomp_mask = mask['decomp']['app']

        assert not ndc_ray
        origins = rays_chunk[:, :3]
        viewdirs = rays_chunk[:, 3:6]

        def sigma_fn(t_starts, t_ends, ray_indices):
            t_origins = origins[ray_indices]
            t_dirs = viewdirs[ray_indices]
            positions = t_origins + t_dirs * (t_starts + t_ends)[..., None] / 2.0
            if t_origins.shape[0] == 0:
                return torch.zeros((0,), device=t_origins.device)
            return (
                self.feature2density(  # type: ignore
                    self.compute_densityfeature(
                        self.normalize_coord(positions),
                        den_decomp_mask
                    )
                )
                * self.distance_scale
            )

        def rgb_sigma_fn(t_starts, t_ends, ray_indices):
            t_origins = origins[ray_indices]
            t_dirs = viewdirs[ray_indices]
            if t_origins.shape[0] == 0:
                return torch.zeros(
                    (0, 3), device=t_origins.device
                ), torch.zeros((0,), device=t_origins.device)
            positions = t_origins + t_dirs * (t_starts + t_ends)[..., None] / 2.0
            positions = self.normalize_coord(positions)
            sigmas = (
                self.feature2density(  # type: ignore
                    self.compute_densityfeature(positions, den_decomp_mask)
                )
                * self.distance_scale
            )
            rgbs = self.renderModule(
                positions, t_dirs, self.compute_appfeature(positions, app_decomp_mask), encoding_mask
            )
            return rgbs, sigmas

        ray_indices, t_starts, t_ends = self.occGrid.sampling(
            origins,
            viewdirs,
            sigma_fn=sigma_fn,
            near_plane=self.near_far[0],
            far_plane=self.near_far[1],
            render_step_size=self.stepSize,
            stratified=is_train,
        )
        rgb_map, _, depth_map, _ = nerfacc.rendering(
            t_starts,
            t_ends,
            ray_indices=ray_indices,
            n_rays=origins.shape[0],
            rgb_sigma_fn=rgb_sigma_fn,
            render_bkgd=1 if white_bg else 0,
        )

        return rgb_map, depth_map, t_starts.shape[0]

    def forward(
        self,
        rays_chunk,
        mask,
        white_bg=True,
        is_train=False,
        ndc_ray=False,
        N_samples=-1,
    ):
        if self.occGrid is not None:
            import nerfacc
            return self._forward_nerfacc(
                rays_chunk, mask, white_bg, is_train, ndc_ray, N_samples
            )
        else:
            return self._forward(
                rays_chunk, mask, white_bg, is_train, ndc_ray, N_samples
            )
