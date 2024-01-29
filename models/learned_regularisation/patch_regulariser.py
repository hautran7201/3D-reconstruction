from dataclasses import dataclass
from dataLoader.ray_utils import get_ray_patch_directions, get_ray_directions
from dataLoader.ray_utils import get_rays as get_rays_utils 
import os
import cv2
import random
import numpy as np
import matplotlib.cm
import matplotlib.pyplot as plt

from typing import Optional, Dict, Tuple
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.functional import grid_sample

from models.learned_regularisation.diffusion.denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer, \
    normalize_to_neg_one_to_one, unnormalize_to_zero_to_one
from models.learned_regularisation.intrinsics import Intrinsics
from models.learned_regularisation.patch_pose_generator import PatchPoseGenerator, FrustumRegulariser
from utils import get_rays, visualize_depth_numpy, visualize_depth


class DepthPreprocessor:
    """
    Preprocesses arrays of depths for feeding into the diffusion model.
    Must be used with the same arguments for both train and test.
    """
    def __init__(self, min_depth: float):
        self.min_depth = min_depth

    def __call__(self, depth, min_depth=None):
        """
        :param depth: Array of depths.
        :returns: Inverse depths, clipped so that the minimum depth is self.min_depth
        """
        """input_min = torch.max(depth)
        if input_min < self.min_depth:
            min_depth = input_min
        else:
            min_depth = self.min_depth"""

        """print()
        print('-----------------------------------------------')
        print(torch.min(depth))
        print(torch.max(depth))"""

        depth = torch.maximum(depth, torch.full_like(input=depth, fill_value=self.min_depth))

        """print(torch.min(depth))
        print(torch.max(depth))"""

        # Depth in range [self._min_depth, inf] -> inv_depth in range [0, 1/self._min_depth]
        inv_depth = 1. / depth
        inv_depth_1 = inv_depth

        # Linearly transform to range [0, 1], just like an rgb channel. The trainer will then transform to [-1, 1].
        inv_depth = inv_depth * self.min_depth
        inv_depth_2 = inv_depth

        return inv_depth, (depth, inv_depth_1, inv_depth_2)

    def invert(self, inv_depth):
        inv_depth = inv_depth / self.min_depth

        depth = 1. / inv_depth

        return depth


# Approximately matches the intrinsics for the FOV dataset
LLFF_DEFAULT_PSEUDO_INTRINSICS = Intrinsics(
    fx=700.,
    fy=700.,
    cx=512.,
    cy=384.,
    width=1024,
    height=768,
)

HUMAN_DEFAULT_PSEUDO_INTRINSICS = Intrinsics(
    fx=1111.1110311937682,
    fy=1111.1110311937682,
    cx=200.,
    cy=200.,
    width=400.,
    height=400.,
)


def make_random_patch_intrinsics(patch_size: int, full_image_intrinsics: Intrinsics,
                                 downscale_factor: int = 1) -> Intrinsics:
    """
    Makes intrinsics corresponding to a random patch sampled from the original image.
    This is required when we want to sample a patch from a training image rather than render one.
    :param patch_size: Size of patch in pixels
    :param full_image_intrinsics: Intrinsics of full original image
    :param downscale_factor: Number of original image pixels per patch pixel. If 1, no downscaling occurs
    :return: Intrinsics for patch as described above
    """

    """print('\n\n\n')
    print(downscale_factor)
    print(full_image_intrinsics)"""

    intrinsics_downscaled = Intrinsics(
        fx=full_image_intrinsics.fx // downscale_factor,
        fy=full_image_intrinsics.fy // downscale_factor,
        cx=full_image_intrinsics.cx // downscale_factor,
        cy=full_image_intrinsics.cy // downscale_factor,
        width=full_image_intrinsics.width // downscale_factor,
        height=full_image_intrinsics.height // downscale_factor,
    )

    """print(
        intrinsics_downscaled.fx,
        intrinsics_downscaled.fy,
        intrinsics_downscaled.cx,
        intrinsics_downscaled.cy,
        intrinsics_downscaled.width,
        intrinsics_downscaled.height,
    )"""

    # print('\n\n\n')

    # Allow our sampled patch to extend past the edges of the img, as long as it overlaps at least marginally with it
    extra_margin = patch_size/2.

    delta_x = intrinsics_downscaled.width - patch_size
    delta_y = intrinsics_downscaled.height - patch_size
    patch_centre_x = random.uniform(intrinsics_downscaled.cx - delta_x - extra_margin,
                                    intrinsics_downscaled.cx + extra_margin)
    patch_centre_y = random.uniform(intrinsics_downscaled.cy - delta_y,
                                    intrinsics_downscaled.cy + extra_margin)

    return Intrinsics(
        fx=intrinsics_downscaled.fx,
        fy=intrinsics_downscaled.fy,
        cx=patch_centre_x,
        cy=patch_centre_y,
        width=patch_size,
        height=patch_size,
    )


def load_patch_diffusion_model(path: Path, device) -> nn.Module:
    """
    Load the patch denoising diffusion model.
    :param path: Path to a checkpoint for the model.
    """
    image_size = 48
    reg_checkpoint_path = path
    channels = 4
    denoising_model = Unet(
        dim=64,
        dim_mults=(1, 2, 4, 8),
        channels=channels,
        self_condition=False,
    ).to(device)

    diffusion = GaussianDiffusion(
        denoising_model,
        image_size=image_size,
        timesteps=1000,  # number of steps
        sampling_timesteps=250,
        loss_type='l1',  # L1 or L2
    ).to(device)
    
    trainer = Trainer(
        diffusion,
        [None],
        train_batch_size=1,
        train_lr=1e-4,
        train_num_steps=700000,  # total training steps
        gradient_accumulate_every=2,  # gradient accumulation steps
        ema_decay=0.995,  # exponential moving average decay
        amp=False,  # turn on mixed precision
        results_folder=reg_checkpoint_path.parent,
        save_and_sample_every=250,
        num_samples=1,
        tb_path=None,
    )
    trainer.load(str(reg_checkpoint_path))
    trainer.ema.ema_model.eval()
    return trainer.ema.ema_model


class DiffusionTimeHandler:
    def __init__(self, diffusion_model: GaussianDiffusion):
        times = torch.linspace(-1, diffusion_model.num_timesteps - 1,
                               steps=diffusion_model.sampling_timesteps + 1)  # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        self._time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

    def get_timesteps(self, time: float) -> Tuple[int, int]:
        """
        :param time: Value of tau, the diffusion time parameter which runs from one to zero during denoising.
        :return: Tuple of (current, next) integer timestep for the diffusion model.
        """
        time = 1. - time
        assert 0. <= time <= 1.
        time_idx = int(round(time * len(self._time_pairs)))
        time_idx = min(time_idx, len(self._time_pairs) - 1)
        time_idx = max(time_idx, 0)

        time, time_next = self._time_pairs[time_idx]
        return time, time_next


@dataclass
class PatchOutputs:
    # Dataclass to store all outputs of a call to the PatchRegulariser.
    images: Dict[str, torch.Tensor]
    loss: torch.Tensor
    rgb_patch: torch.Tensor
    depth_patch: torch.Tensor
    disp_patch: torch.Tensor
    render_outputs: Dict


DISPARITY_CMAP = matplotlib.cm.get_cmap('plasma')
DEPTH_CMAP = matplotlib.cm.get_cmap('plasma')


class PatchRegulariser:
    """
    Main class for using the denoising diffusion patch model to regularise NeRFs.
    """
    def __init__(
        self, 
        pose_generator: PatchPoseGenerator, 
        patch_diffusion_model: GaussianDiffusion,
        full_image_intrinsics: Intrinsics, 
        device,
        planar_depths: bool, 
        frustum_regulariser: Optional[FrustumRegulariser], 
        patch_size: int = 48,
        image_sample_prob: float = 0.,
        uniform_in_depth_space: bool = False,
        sample_downscale_factor: int = 4
    ):
        """
        :param pose_generator: PatchPoseGenerator which will be used to provide camera poses from which to render
            patches.
        :param patch_diffusion_model: Denoising diffusion model to use as the score function for regularisation.
        :param full_image_intrinsics: Intrinsics for the training images.
        :param device: Torch device to do calculations on.
        :param planar_depths: Should be true if the diffusion model was trained using depths projected along the z-axis
            rather than Cartesian distances.
        :param frustum_regulariser: Frustum regulariser instance, or None of the frustum loss is not desired.
        :param image_sample_prob: Fraction of the time to sample a patch from a training image directly, rather than
            rendering.
        :param uniform_in_depth_space: If True, losses will be normalised w.r.t. the depth as described in the paper.
        :param sample_downscale_factor: Downscale factor to apply before sampling. This will allow the patch to
            correspond to a wider FOV.
        """
        self._pose_generator = pose_generator
        self._device = device
        self._diffusion_model = patch_diffusion_model.to(self._device)
        self._planar_depths = planar_depths
        self._image_sample_prob = image_sample_prob
        self.frustum_regulariser = frustum_regulariser
        self._uniform_in_depth_space = uniform_in_depth_space
        self._sample_downscale_factor = sample_downscale_factor
        self._depth_preprocessor = DepthPreprocessor(min_depth=2.8)
        self._patch_size = patch_size
        self._full_image_intrinsics = full_image_intrinsics
        self._time_handler = DiffusionTimeHandler(diffusion_model=self._diffusion_model)

        print('Num channels in diffusion model:', self._diffusion_model.channels)

    def get_diffusion_loss_with_rendered_patch(self, model, time) -> PatchOutputs:
        depth_patch, rgb_patch, render_outputs = self._render_random_patch(model)

        """if iteration > 0 and iteration % train_every == 0:
            # ####################################
            t_depth_patch = depth_patch.squeeze().cpu().numpy()
            t_rgb_patch = rgb_patch.squeeze().cpu().numpy()        

            # Hình ảnh màu RGB
            fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(15, 20))
            ax[0][0].imshow(t_depth_patch)            
            ax[0][1].imshow(t_rgb_patch)            

            # Hiển thị hình ảnh
            savefig = fig.savefig(f"/content/test/{iteration}.png")
            plt.close()
            # ####################################"""

        return self.get_loss_for_patch(depth_patch=depth_patch, rgb_patch=rgb_patch, time=time,
                                       render_outputs=render_outputs, file_name='get_diffusion_loss_with_rendered_patch')

    def get_diffusion_loss_with_sampled_patch(self, model, time, image, image_intrinsics,
                                              pose, iteration=-1, train_every=0) -> PatchOutputs:
        # As described in the paper, we sometimes sample a patch from the image rather than rendering it using the NeRF.
        # This function does that (though we still have to render the depth channel using the NeRF).
        while True:
            depth_patch, rgb_patch, render_outputs = self._sample_patch(
                image=image, image_intrinsics=image_intrinsics, pose=pose, model=model
            )

            """if iteration > 0 and iteration % train_every == 0:
                # ####################################
                t_depth_patch = depth_patch.squeeze().cpu().numpy()
                t_rgb_patch = rgb_patch.squeeze().cpu().numpy()        

                # Hình ảnh màu RGB
                fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(15, 20))
                ax[0][0].imshow(t_depth_patch)            
                ax[0][1].imshow(t_rgb_patch)            

                # Hiển thị hình ảnh
                savefig = fig.savefig(f"/content/test/{iteration}.png")
                plt.close()
                exit()
                # ####################################"""

            return self.get_loss_for_patch(depth_patch=depth_patch, rgb_patch=rgb_patch, time=time,
                                            update_depth_only=True, render_outputs=render_outputs, file_name='get_diffusion_loss_with_sampled_patch')                                         
            try:
                depth_patch, rgb_patch, render_outputs = self._sample_patch(
                    image=image, image_intrinsics=image_intrinsics, pose=pose, model=model
                )
                return self.get_loss_for_patch(depth_patch=depth_patch, rgb_patch=rgb_patch, time=time,
                                               update_depth_only=True, render_outputs=render_outputs, file_name='get_diffusion_loss_with_sampled_patch')
            except AssertionError as e:
                print('Exception:', str(e))
                exit()

    def get_loss_for_patch(self, depth_patch, rgb_patch, time, render_outputs,
                           update_depth_only: bool = False, file_name='') -> PatchOutputs:
        disparity_patch, all_depth = self._depth_preprocessor(depth_patch) # .squeeze().cpu().numpy()
        disparity_patch = disparity_patch.squeeze().cpu().numpy()
        _disparity_patch, _ = visualize_depth(depth_patch.squeeze().cpu().numpy(), [2, 6])

        if False:
            import matplotlib.pyplot as plt

            #####################################
            """print('[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]')
            print(disparity_patch.shape)
            print('max min')
            ma = np.max(disparity_patch)
            mi = np.min(disparity_patch)
            print(ma)
            print(mi)"""
          
            disparity_patch
            # Vẽ hình ảnh

            # Hình ảnh màu RGB
            fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(15, 20))
            # ax[0][0].imshow(t_image_rgb)            
            ax[0][0].imshow(_disparity_patch, cmap='gray')            
            ax[0][0].set_title('t_disparity_patch')
            ax[0][1].imshow(disparity_patch, cmap='gray')            
            ax[0][1].set_title('_disparity_patch')

            path = '/content/test/'
            savefig = fig.savefig(f"{path}{file_name}.png")
            plt.close()

            # disparity_patch = (disparity_patch - mi) / (ma - mi + 1e-8)

        time, time_next = self._time_handler.get_timesteps(time)

        disparity_patch = torch.Tensor(disparity_patch)[None, ..., None].to(self._device)
        # Possibly regularise RGB too, if the model has that
        if self._diffusion_model.channels == 4:            
            device = disparity_patch.device
            patch = torch.cat([rgb_patch.to(device), disparity_patch], dim=-1)
        elif self._diffusion_model.channels == 1:
            patch = disparity_patch
        else:
            raise ValueError('Diffusion model must have 1 channel (D) or 4 channels (RGBD)')

        # Go from [B, H, W, C] to [B, C, H, W]
        patch = patch.moveaxis(-1, 1)

        patch = normalize_to_neg_one_to_one(patch)

        model_predictions = self._diffusion_model.model_predictions(
            x=patch,
            t=torch.Tensor([time], ).to(torch.int64).to(self._device),
            clip_x_start=True
        )

        sigma_lambda = (1. - self._diffusion_model.alphas_cumprod[time]).sqrt()
        assert sigma_lambda > 0.
        grad_log_prior_prob = -model_predictions.pred_noise.detach() * (1. / sigma_lambda)
        if grad_log_prior_prob.isfinite().all() == False:
            print('\n\n-----------------all_depth-------------\n\n')
            print(torch.isnan(depth_patch).any().item())
            print(torch.isnan(all_depth[0]).any().item())
            print(torch.isnan(all_depth[1]).any().item())
            print(torch.isnan(all_depth[2]).any().item())
            print()
            print()
            print(grad_log_prior_prob.isfinite().all())
            print()
            print()
            print('NAN')
            print(torch.isnan(rgb_patch).any().item())
            print(torch.isnan(disparity_patch).any().item())
            print()
            print()
            print('patch')
            print(patch)
            print()
            print()
            print('torch.Tensor([time], ).to(torch.int64).to(self._device)')
            print(torch.Tensor([time], ).to(torch.int64).to(self._device))
            print()
            print()
            print('self._diffusion_model.channels')
            print(self._diffusion_model.channels)
            exit()

        # Multipliers so that the input weight parameters for depth and rgb can be kept close to unity for convenience
        depth_weight = 2e-6
        rgb_weight = 1e-9 if not update_depth_only else 0.

        # NOTE: below we compute a loss L = -(constant * patch * grad log P).
        # This is done so that dL/d(patch) = -constant * grad log P, so that we are injecting grad log P into
        #   the gradients while we fit our nerfs.

        # First calculate the loss for the depth channel
        # If requested, we also normalise by multiplying by the inverse depth (i.e. dividing by the depth),
        #   as described in the supplemental.
        if self._uniform_in_depth_space:
            depth_patch_detached = depth_patch.moveaxis(-1, 1).detach()
            normalisation_const = 1.
            multiplier = depth_patch_detached
            assert multiplier.isfinite().all()
            diffusion_pseudo_loss = -torch.sum(depth_weight * multiplier * grad_log_prior_prob[:, -1, :, :] * patch[:, -1, :, :])
        else:
            diffusion_pseudo_loss = -torch.sum(depth_weight * grad_log_prior_prob[:, -1, :, :] * patch[:, -1, :, :])

        # 4-channel models are assumed to be RGBD:
        if self._diffusion_model.channels == 4:
            normalisation_const = 3000
            multiplier = normalisation_const / torch.linalg.norm(grad_log_prior_prob[:, :-1, :, :].detach())
            diffusion_pseudo_loss += -torch.sum(multiplier * rgb_weight * grad_log_prior_prob[:, :-1, :, :] * patch[:, :-1, :, :])

        # print(grad_log_prior_prob)
        # assert grad_log_prior_prob.isfinite().all()

        pred_noise_bhwc = unnormalize_to_zero_to_one(torch.moveaxis(model_predictions.pred_noise, 1, -1))
        pred_x0_bhwc = unnormalize_to_zero_to_one(torch.moveaxis(model_predictions.pred_x_start, 1, -1))

        patch_outputs = PatchOutputs(
            images={
                'rendered_rgb': rgb_patch,
                'rendered_depth': depth_patch,
                'rendered_disp': disparity_patch,
                'pred_disp_noise': pred_noise_bhwc[..., -1],
                'pred_disp_x0': pred_x0_bhwc[..., -1],
                'pred_depth_x0': self._depth_preprocessor.invert(pred_x0_bhwc[..., -1]),
            },
            loss=diffusion_pseudo_loss,
            depth_patch=depth_patch,
            disp_patch=disparity_patch,
            rgb_patch=rgb_patch,
            render_outputs=render_outputs,
        )
        # Also compute 'what would the patch look like if we took a step in the
        #   direction that the diffusion model wants to go?'
        # The scale factor which multiplies the step below is basically arbitrary since this is just
        #   for visualisation purposes.
        step_scale_factor = 5e-4
        patch_outputs.images['disp_plus_step'] = patch_outputs.images['rendered_disp'] - \
            patch_outputs.images['pred_disp_noise'].unsqueeze(-1) * step_scale_factor / sigma_lambda

        if self._diffusion_model.channels == 4:
            patch_outputs.images['pred_rgb_noise'] = pred_noise_bhwc[..., :-1]
            patch_outputs.images['pred_rgb_x0'] = pred_x0_bhwc[..., :-1]

        return patch_outputs

    def _render_random_patch(self, model):
        pose = self._pose_generator.generate_random_with_sphere() # 
        intrinsics = self._get_random_patch_intrinsics()
        pred_depth, pred_rgb, _, render_outputs = self._render_patch_with_intrinsics(intrinsics=intrinsics,
                                                                                     pose=pose, model=model)
        # print('[[[[[[[[[[[[[[[[[[[[[[[[[[_render_random_patch]]]]]]]]]]]]]]]]]]]]]]]]]]')
        if False: 
            path = '/content/test/'
            isExist = os.path.exists(path)
            if not isExist:
              os.makedirs(path)

            import matplotlib.pyplot as plt

            #####################################
            t_image_grayscale = pred_depth.squeeze().cpu().numpy()
            t_rendered_rgb = pred_rgb.squeeze().cpu().detach().numpy()          
            # Vẽ hình ảnh

            # Hình ảnh màu RGB
            fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(15, 20))
            ax[0][0].imshow(t_rendered_rgb, cmap='gray')
            ax[0][0].set_title('t_image_rgb')
            ax[0][1].imshow(t_image_grayscale, cmap='gray')
            ax[0][1].set_title('t_image_grayscale')            

            # Hiển thị hình ảnh
            savefig = fig.savefig(f"{path}test_render_random_patch.png")
            plt.close()

        return pred_depth, pred_rgb, render_outputs

    def _render_patch_with_intrinsics(self, intrinsics, pose, model):
        pseudo_intrinsics = (intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy)

        ##############################
        if False:
            # directions = get_ray_directions(self._patch_size, self._patch_size, [intrinsics.fx, intrinsics.fy])
            directions = get_ray_patch_directions(poses=pose.unsqueeze(0), intrinsics=pseudo_intrinsics,
                                  H=self._patch_size, W=self._patch_size, N=-1)
            
            directions = directions / torch.norm(directions, dim=-1, keepdim=True)
            rays_d_cam = directions.view(1, -1, 3)
            rays_d_cam_z = rays_d_cam[..., -1]
            rays_o, rays_d = get_rays_utils(directions, pose)  # both (h*w, 3)
            patch_rays = {
                'rays_o': rays_o[None, ...],
                'rays_d': rays_d[None, ...],
                'rays_d_cam': rays_d_cam,
                'rays_d_cam_z': rays_d_cam_z
            }
        else:
            patch_rays = get_rays(poses=pose.unsqueeze(0), intrinsics=pseudo_intrinsics,
                                  H=self._patch_size, W=self._patch_size, N=-1)

        ray_o = patch_rays['rays_o'].squeeze(0)
        ray_d = patch_rays['rays_d'].squeeze(0)
        rays = torch.cat((ray_o, ray_d), 1)

        outputs = model.render(rays, white_bg=False, is_train=False)

        type_map = 'depth' # disparity
        if self._planar_depths:
            device = outputs[type_map].device 
            depth = outputs[type_map] * patch_rays['rays_d_cam_z'].to(device)
        else:
            depth = outputs[type_map]

        B = 1
        pred_depth = depth.reshape(B, intrinsics.height, intrinsics.width, 1)
        pred_rgb = outputs['rgb_map'].reshape(B, intrinsics.height, intrinsics.width, 3)

        return pred_depth, pred_rgb, patch_rays, outputs

    def _sample_patch(self, image, image_intrinsics, pose, model):
        patch_intrinsics = self._get_random_patch_intrinsics()

        rendered_depth, rendered_rgb, patch_rays, render_outputs = self._render_patch_with_intrinsics(
            intrinsics=patch_intrinsics, pose=pose, model=model
        )
        with torch.no_grad():
            gt_rgb = sample_patch_from_img(rays_d=patch_rays['rays_d_cam'], img=image,
                                           img_intrinsics=image_intrinsics, patch_size=self._patch_size)

        # rendered_rgb = rendered_rgb.flip(1).flip(2)
        # rendered_depth = rendered_depth.flip(1).flip(2) 
        gt_rgb = gt_rgb.flip(1).flip(2) 
        # print('[[[[[[[[[[[[[[[[[[[[[[[[[[_sample_patch]]]]]]]]]]]]]]]]]]]]]]]]]]')
        if False: 
            path = '/content/test/'
            isExist = os.path.exists(path)
            if not isExist:
              os.makedirs(path)

            import matplotlib.pyplot as plt

            #####################################
            t_image_rgb = gt_rgb.squeeze().cpu().numpy()
            t_image_grayscale = rendered_depth.squeeze().cpu().numpy()
            t_rendered_rgb = rendered_rgb.squeeze().cpu().detach().numpy()          
            disparity = render_outputs['disparity'].cpu().detach().numpy().reshape(48, 48, 1)
            depth = render_outputs['depth'].cpu().detach().numpy().reshape(48, 48)
            depth_final, _ = visualize_depth_numpy(depth, [2, 6])
            depth_final1, _ = visualize_depth(depth)
            # Vẽ hình ảnh

            # Hình ảnh màu RGB
            fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(15, 20))
            # ax[0][0].imshow(t_image_rgb)            
            ax[0][0].imshow(depth, cmap='gray')            
            ax[0][0].set_title('t_image_rgb')
            # ax[0][1].imshow(t_image_grayscale)            
            ax[0][1].imshow(depth_final1, cmap='gray')            
            ax[0][1].set_title('t_image_grayscale')
            ax[1][0].imshow(image) 
            ax[1][0].set_title('image')
            ax[1][1].imshow(t_rendered_rgb) 
            ax[1][1].set_title('t_rendered_rgb')

            # Hiển thị hình ảnh
            savefig = fig.savefig(f"{path}test_sample_patch.png")
            plt.close()
            #####################################

        # return rendered_depth, gt_rgb, render_outputs
        return rendered_depth, gt_rgb, render_outputs

    def _get_random_patch_intrinsics(self) -> Intrinsics:
        return make_random_patch_intrinsics(
            patch_size=self._patch_size,
            full_image_intrinsics=self._full_image_intrinsics,
            downscale_factor=self._sample_downscale_factor,
        )

    def dump_debug_visualisations(self, output_folder: Path, output_prefix: str, patch_outputs: PatchOutputs) -> None:
        disp_keys = ('pred_disp_x0', 'rendered_disp', 'disp_plus_step')
        depth_keys = ('pred_depth_x0', 'rendered_depth')
        for key_set in (disp_keys, depth_keys):
            keys_present = [k for k in patch_outputs.images if k in key_set]
            imgs = normalise_together([patch_outputs.images[k] for k in keys_present])
            for k, img_normed in zip(keys_present, imgs):
                patch_outputs.images[k] = img_normed

        for k, img in patch_outputs.images.items():
            """print('key', k)
            print('img shape', img.shape)"""
            img = img[0]
            img = img.squeeze(dim=-1)
            img = img.detach().cpu().numpy()

            if 'disp' in k:
                img = DISPARITY_CMAP(img)
            elif 'depth' in k:
                img = DEPTH_CMAP(img)

            image_path = output_folder / f'{output_prefix}-{k}.png'
            cv2.imwrite(str(image_path), cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))


def normalise_together(imgs):
    max_val = max(img.max() for img in imgs)
    return [img/max_val for img in imgs]


def sample_patch_from_img(rays_d, img, img_intrinsics, patch_size: int):
    """
    Sample a patch from an image.
    :param rays_d: rays_d as returned by the get_rays() function
    :param img: Image to sample the patch from from
    :param img_intrinsics: Intrinsics for the image as a four-element sequence (fx, fy, cx, cy) in pixel coords.
    :param patch_size: Side length of the patch to make, in units of pixels
    """
    fx, fy, cx, cy = img_intrinsics
    h, w, c = img.shape    

    b, num_rays, n_dim = rays_d.shape
    assert n_dim == 3
    assert b == 1

    # Go from ray directions in camera frame to pixel coordinates
    pixel_i = fx/2 * rays_d[..., 0] / rays_d[..., 2] + cx + 0.5
    pixel_j = fy/2 * rays_d[..., 1] / rays_d[..., 2] + cy + 0.5

    assert (pixel_i >= 0.).all()
    assert (pixel_j >= 0.).all()

    assert (pixel_i <= w).all()
    assert (pixel_j <= h).all()

    # Normalise - grid_sample wants query locations in [-1, 1].
    pixel_i = pixel_i / w
    pixel_j = pixel_j / h
    pixel_i = 2. * pixel_i - 1.
    pixel_j = 2. * pixel_j - 1.

    pixel_coords = torch.cat([pixel_i.unsqueeze(-1).unsqueeze(-1), pixel_j.unsqueeze(-1).unsqueeze(-1)], dim=-1)
    img_reshaped = img.moveaxis(-1, 0).unsqueeze(0)
    sampled = grid_sample(input=img_reshaped, grid=pixel_coords, align_corners=True)

    sampled = sampled.reshape(1, c, patch_size, patch_size)
    sampled = sampled.moveaxis(1, -1).flip(1).flip(2)

    return sampled