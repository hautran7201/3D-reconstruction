import torch,os,imageio,sys
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from dataLoader.ray_utils import get_rays
from models.tensoRF import TensorVM, TensorCP, raw2alpha, TensorVMSplit, AlphaGridMask
from utils import *
from dataLoader.ray_utils import ndc_rays_blender


def OctreeRender_trilinear_fast(rays, tensorf, mask=None, chunk=4096, N_samples=-1, ndc_ray=False, white_bg=True, is_train=False, device='cuda'):

    rgbs, alphas, depth_maps, weights, uncertainties, num_samples = [], [], [], [], [], []
    N_rays_all = rays.shape[0]
    for chunk_idx in range(N_rays_all // chunk + int(N_rays_all % chunk > 0)):
        rays_chunk = rays[chunk_idx * chunk:(chunk_idx + 1) * chunk].to(device)
    
        rgb_map, depth_map, num_valid_samples = tensorf(rays_chunk, mask, is_train=is_train, white_bg=white_bg, ndc_ray=ndc_ray, N_samples=N_samples)

        rgbs.append(rgb_map)
        depth_maps.append(depth_map)
        num_samples.append(float(num_valid_samples))

    return torch.cat(rgbs), None, torch.cat(depth_maps), None, None, sum(num_samples)


def create_gif(path_to_dir, name_gif):
    if os.path.exists(path_to_dir):
        filenames = os.listdir(path_to_dir)
        filenames = sorted(filenames, key=lambda x: int(x.split('.')[0]))
        images = []
        for filename in filenames:
            images.append(imageio.imread(f'{path_to_dir}/{filename}'))
        kargs = {"duration": 5.0}
        imageio.mimsave(name_gif, images, "GIF", **kargs)
    else:
        return

@torch.no_grad()
def PSNRs_calculate(args, tensorf, dataset, renderer, chunk=4096, N_samples=-1,
               white_bg=False, ndc_ray=False, compute_extra_metrics=False, device='cuda'):
    PSNRs, rgb_maps, depth_maps = [], [], []
    ssims,l_alex,l_vgg=[],[],[]

    try:
        tqdm._instances.clear()
    except Exception:
        pass

    near_far = dataset.near_far
    img_eval_interval = 1 
    idxs = list(range(0, dataset.all_rays.shape[0], img_eval_interval))
    for idx, samples in tqdm(enumerate(dataset.all_rays[0::img_eval_interval]), file=sys.stdout):

        W, H = dataset.img_wh
        rays = samples.to(device).view(-1,samples.shape[-1])

        rgb_map, _, depth_map, _, _, _ = renderer(
            rays,
            tensorf,
            chunk=chunk,
            N_samples=N_samples,
            ndc_ray=ndc_ray,
            white_bg=white_bg,
            device=device,
        )
        
        rgb_map = rgb_map.clamp(0.0, 1.0)

        rgb_map, depth_map = rgb_map.reshape(H, W, 3).cpu(), depth_map.reshape(H, W).cpu()

        depth_map, _ = visualize_depth_numpy(depth_map.numpy(),near_far)
        if len(dataset.all_rgbs):
            gt_rgb = dataset.all_rgbs[idxs[idx]].view(H, W, 3)
            loss = torch.mean((rgb_map - gt_rgb) ** 2)
            PSNRs.append(-10.0 * np.log(loss.item()) / np.log(10.0))

            if compute_extra_metrics:
                ssim = rgb_ssim(rgb_map, gt_rgb, 1)
                l_a = rgb_lpips(gt_rgb.numpy(), rgb_map.numpy(), 'alex', tensorf.device)
                l_v = rgb_lpips(gt_rgb.numpy(), rgb_map.numpy(), 'vgg', tensorf.device)
                ssims.append(ssim)
                l_alex.append(l_a)
                l_vgg.append(l_v)

    return PSNRs

@torch.no_grad()
def save_rendered_image_per_train(train_dataset, test_dataset, tensorf, renderer, step, logs, savePath=None, chunk=4096, N_samples=-1,
               white_bg=False, ndc_ray=False, compute_extra_metrics=True, device='cuda'):
    PSNRs, rgb_maps, depth_maps = [], [], []
    ssims, l_alex, l_vgg        = [], [], []
    os.makedirs(savePath, exist_ok=True)
    os.makedirs(savePath+"/rgb", exist_ok=True)
    os.makedirs(savePath+"/rgbd", exist_ok=True)
    os.makedirs(savePath+f"/plot/vis_every", exist_ok=True)

    try:
        tqdm._instances.clear()
    except Exception:
        pass

    near_far          = train_dataset.near_far
    idxs              = list(range(0, train_dataset.all_rays.shape[0], 1))
    train_rgb_map     = None
    train_depth_map   = None
    img_eval_interval = 1
    for idx, samples in enumerate(train_dataset.all_rays[0::img_eval_interval]):

        W, H = train_dataset.img_wh
        rays = samples.view(-1,samples.shape[-1])

        rgb_map, _, depth_map, _, _, _ = renderer(
            rays,
            tensorf,
            chunk=chunk,
            N_samples=N_samples,
            ndc_ray=ndc_ray,
            white_bg=white_bg,
            device=device,
        )

        rgb_map             = rgb_map.clamp(0.0, 1.0)

        rgb_map, depth_map   = rgb_map.reshape(H, W, 3).cpu(), depth_map.reshape(H, W).cpu()

        train_depth_map, _  = visualize_depth_numpy(depth_map.numpy(),near_far)

        train_rgb_map       = (rgb_map.numpy() * 255).astype('uint8')



    near_far = test_dataset.near_far
    img_eval_interval = 1 
    idxs = list(range(0, test_dataset.all_rays.shape[0], img_eval_interval))
    test_rgb_map = None
    test_depth_map = None

    for idx, samples in enumerate(test_dataset.all_rays[0::img_eval_interval]):

        W, H = test_dataset.img_wh
        rays = samples.view(-1,samples.shape[-1])

        rgb_map, _, depth_map, _, _, _ = renderer(
            rays,
            tensorf,
            chunk=chunk,
            N_samples=N_samples,
            ndc_ray=ndc_ray,
            white_bg=white_bg,
            device=device,
        )

        rgb_map = rgb_map.clamp(0.0, 1.0)

        rgb_map, depth_map = rgb_map.reshape(H, W, 3).cpu(), depth_map.reshape(H, W).cpu()

        test_depth_map, _ = visualize_depth_numpy(depth_map.numpy(),near_far)

        test_rgb_map = (rgb_map.numpy() * 255).astype('uint8')
        # rgb_map = np.concatenate((rgb_map, depth_map), axis=1)


    if savePath is not None:
        loss = logs["mse"]
        train_psnr = logs["train_psnr"]
        test_psnr = logs["test_psnr"]
        
        # Plot the rgb, depth and the loss plot.
        fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(15, 20))
        ax[0][0].imshow(train_rgb_map)
        ax[0][0].set_title(f"Predicted train Image: {step:03d}")

        ax[0][1].imshow(train_depth_map)
        ax[0][1].set_title(f"Train Image with Depth Map: {step:03d}")

        ax[1][0].imshow(test_rgb_map)
        ax[1][0].set_title(f"Predicted test Image: {step:03d}")

        ax[1][1].imshow(test_depth_map)
        ax[1][1].set_title(f"Test Image with Depth Map: {step:03d}")


        W, H = train_dataset.img_wh
        ax[2][0].plot(loss)
        ax[2][0].set_title(f"Loss Plot: {step:03d}")
        ax[2][0].set_box_aspect(H/W)

        ax[2][1].plot(train_psnr, label='Train')
        ax[2][1].plot(test_psnr, label='Test')
        ax[2][1].set_title(f"Train test psnr Plot: {step:03d}")
        ax[2][1].set_box_aspect(H/W)
        
        savefig = fig.savefig(f"{savePath}/plot/vis_every/{step:03d}.png")
        plt.close()

@torch.no_grad()
def evaluation(test_dataset,tensorf, args, renderer, savePath=None, N_vis=5, prtx='', chunk=4096, N_samples=-1,
               white_bg=False, ndc_ray=False, compute_extra_metrics=True, device='cuda'):
    PSNRs, gt_rgb_maps, rgb_maps, depth_maps = [], [], [], []
    ssims,l_alex,l_vgg=[],[],[]
    os.makedirs(savePath, exist_ok=True)
    os.makedirs(savePath+"/rgbd", exist_ok=True)
    os.makedirs(savePath+"/prediction", exist_ok=True)
    os.makedirs(savePath+"/ground_truth", exist_ok=True)

    try:
        tqdm._instances.clear()
    except Exception:
        pass

    near_far = test_dataset.near_far
    img_eval_interval = 1 if N_vis < 0 else max(test_dataset.all_rays.shape[0] // N_vis,1)
    idxs = list(range(0, test_dataset.all_rays.shape[0], img_eval_interval))
    for idx, samples in tqdm(enumerate(test_dataset.all_rays[0::img_eval_interval]), file=sys.stdout):

        W, H = test_dataset.img_wh
        rays = samples.view(-1,samples.shape[-1])

        rgb_map, _, depth_map, _, _, _ = renderer(
            rays,
            tensorf,
            chunk=chunk,
            N_samples=N_samples,
            ndc_ray=ndc_ray,
            white_bg=white_bg,
            device=device,
        )

        rgb_map = rgb_map.clamp(0.0, 1.0)

        rgb_map, depth_map = rgb_map.reshape(H, W, 3).cpu(), depth_map.reshape(H, W).cpu()

        depth_map, _ = visualize_depth_numpy(depth_map.numpy(),near_far)
        if len(test_dataset.all_rgbs):
            gt_rgb = test_dataset.all_rgbs[idxs[idx]].view(H, W, 3)
            loss = torch.mean((rgb_map - gt_rgb) ** 2)
            PSNRs.append(-10.0 * np.log(loss.item()) / np.log(10.0))

            if compute_extra_metrics:
                ssim = rgb_ssim(rgb_map, gt_rgb, 1)
                l_a = rgb_lpips(gt_rgb.numpy(), rgb_map.numpy(), 'alex', tensorf.device)
                l_v = rgb_lpips(gt_rgb.numpy(), rgb_map.numpy(), 'vgg', tensorf.device)
                ssims.append(ssim)
                l_alex.append(l_a)
                l_vgg.append(l_v)

        rgb_map = (rgb_map.numpy() * 255).astype('uint8')
        gt_rgb_map = (gt_rgb.numpy() * 255).astype('uint8')

        rgb_maps.append(rgb_map)
        gt_rgb_maps.append(gt_rgb_map)
        depth_maps.append(depth_map)
        if savePath is not None:
            imageio.imwrite(f'{savePath}/prediction/{prtx}{idx:03d}.png', rgb_map)
            imageio.imwrite(f'{savePath}/ground_truth/{prtx}{idx:03d}.png', gt_rgb_map)
            rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
            imageio.imwrite(f'{savePath}/rgbd/{prtx}{idx:03d}.png', rgb_map)

    imageio.mimwrite(f'{savePath}/{prtx}video.mp4', np.stack(rgb_maps), fps=30, quality=10)
    imageio.mimwrite(f'{savePath}/{prtx}depthvideo.mp4', np.stack(depth_maps), fps=30, quality=10)

    if PSNRs:
        psnr = np.mean(np.asarray(PSNRs))
        if compute_extra_metrics:
            ssim = np.mean(np.asarray(ssims))
            l_a = np.mean(np.asarray(l_alex))
            l_v = np.mean(np.asarray(l_vgg))
            np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr, ssim, l_a, l_v]))
        else:
            np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr]))


    return PSNRs

@torch.no_grad()
def evaluation_path(test_dataset,tensorf, c2ws, renderer, savePath=None, N_vis=5, prtx='', N_samples=-1,
                    white_bg=False, ndc_ray=False, compute_extra_metrics=True, device='cuda'):
    PSNRs, rgb_maps, depth_maps = [], [], []
    ssims,l_alex,l_vgg=[],[],[]
    os.makedirs(savePath, exist_ok=True)
    os.makedirs(savePath+"/rgbd", exist_ok=True)

    try:
        tqdm._instances.clear()
    except Exception:
        pass

    near_far = test_dataset.near_far
    for idx, c2w in tqdm(enumerate(c2ws)):

        W, H = test_dataset.img_wh

        c2w = torch.FloatTensor(c2w)
        rays_o, rays_d = get_rays(test_dataset.directions, c2w)  # both (h*w, 3)
        if ndc_ray:
            rays_o, rays_d = ndc_rays_blender(H, W, test_dataset.focal[0], 1.0, rays_o, rays_d)
        rays = torch.cat([rays_o, rays_d], 1)  # (h*w, 6)

        rgb_map, _, depth_map, _, _, _ = renderer(rays, tensorf, chunk=8192, N_samples=N_samples,
                                        ndc_ray=ndc_ray, white_bg = white_bg, device=device)
        rgb_map = rgb_map.clamp(0.0, 1.0)

        rgb_map, depth_map = rgb_map.reshape(H, W, 3).cpu(), depth_map.reshape(H, W).cpu()

        depth_map, _ = visualize_depth_numpy(depth_map.numpy(),near_far)

        rgb_map = (rgb_map.numpy() * 255).astype('uint8')
        # rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
        rgb_maps.append(rgb_map)
        depth_maps.append(depth_map)
        if savePath is not None:
            imageio.imwrite(f'{savePath}/{prtx}{idx:03d}.png', rgb_map)
            rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
            imageio.imwrite(f'{savePath}/rgbd/{prtx}{idx:03d}.png', rgb_map)

    imageio.mimwrite(f'{savePath}/{prtx}video.mp4', np.stack(rgb_maps), fps=30, quality=8)
    imageio.mimwrite(f'{savePath}/{prtx}depthvideo.mp4', np.stack(depth_maps), fps=30, quality=8)

    if PSNRs:
        psnr = np.mean(np.asarray(PSNRs))
        if compute_extra_metrics:
            ssim = np.mean(np.asarray(ssims))
            l_a = np.mean(np.asarray(l_alex))
            l_v = np.mean(np.asarray(l_vgg))
            np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr, ssim, l_a, l_v]))
        else:
            np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr]))


    return PSNRs