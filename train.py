import os
from tqdm.auto import tqdm

import json, random
from renderer import *
from utils import *
from torch.utils.tensorboard import SummaryWriter
import datetime
from dataLoader import dataset_dict
import sys
 
from models import nerf_math
from opt import config_parser
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import numpy as np
from dataLoader import ray_utils
from models import nerf_math
import timeit

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
renderer = OctreeRender_trilinear_fast

class SimpleSampler:
    def __init__(self, total, batch):
        self.total = total
        self.batch = batch
        self.curr = total
        self.ids = None

    def nextids(self):
        self.curr+=self.batch
        if self.curr + self.batch > self.total:
            self.ids = torch.LongTensor(np.random.permutation(self.total))
            self.curr = 0
        return self.ids[self.curr:self.curr+self.batch]


@torch.no_grad()
def export_mesh(args, ckpt_path):

    print(ckpt_path)
    # ckpt = torch.load(args.ckpt, map_location=device)
    ckpt = torch.load(ckpt_path, map_location=device)
    kwargs = ckpt['kwargs']
    # kwargs.update({'device': device})
    print(args.model_name)
    # tensorf = eval(args.model_name)(args=args, **kwargs)
    tensorf = eval(args.model_name)(**kwargs)
    tensorf.load(ckpt)

    alpha,_ = tensorf.getDenseAlpha()
    convert_sdf_samples_to_ply(alpha.cpu(), f'{ckpt_path[:-3]}.ply',bbox=tensorf.aabb.cpu(), level=0.005)


@torch.no_grad()
def render_test(args):
    # init dataset
    dataset = dataset_dict[args.dataset_name]
    test_dataset = dataset(args.datadir, split='test', downsample=args.downsample_train, is_stack=True)
    white_bg = test_dataset.white_bg
    ndc_ray = args.ndc_ray

    if not os.path.exists(args.ckpt):
        print('the ckpt path does not exists!!')
        return

    ckpt = torch.load(args.ckpt, map_location=device)
    kwargs = ckpt['kwargs']
    kwargs.update({'device': device})
    tensorf = eval(args.model_name)(**kwargs)
    tensorf.load(ckpt)

    logfolder = os.path.dirname(args.ckpt)
    if args.render_train:
        os.makedirs(f'{logfolder}/imgs_train_all', exist_ok=True)
        train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=True)
        PSNRs_test = evaluation(train_dataset,tensorf, args, renderer, f'{logfolder}/imgs_train_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)
        print(f'======> {args.expname} train all psnr: {np.mean(PSNRs_test)} <========================')

    if args.render_test:
        os.makedirs(f'{logfolder}/{args.expname}/imgs_test_all', exist_ok=True)
        evaluation(test_dataset,tensorf, args, renderer, f'{logfolder}/{args.expname}/imgs_test_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)

    if args.render_path:
        c2ws = test_dataset.render_path
        os.makedirs(f'{logfolder}/{args.expname}/imgs_path_all', exist_ok=True)
        evaluation_path(test_dataset,tensorf, c2ws, renderer, f'{logfolder}/{args.expname}/imgs_path_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)

def reconstruction(args):

    # init dataset
    print('========================= LOAD DATASET =========================')    
    print('Dataset name:', args.dataset_name)
    dataset = dataset_dict[args.dataset_name]

    idxs = [26, 86, 2, 55, 75, 16, 73, 8]
    train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=False, N_imgs=args.N_train_imgs, indexs=idxs)
    stack_train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=True, tqdm=False, N_imgs=args.N_train_imgs, indexs=idxs)
    test_dataset = dataset(args.datadir, split='test', downsample=args.downsample_train, is_stack=True, N_imgs=args.N_test_imgs)
    print('\n\n')

    train_visual = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=True, tqdm=False, N_imgs=1)
    test_visual = dataset(args.datadir, split='test', downsample=args.downsample_train, is_stack=True, tqdm=False, N_imgs=1)

    white_bg = train_dataset.white_bg
    near_far = train_dataset.near_far
    ndc_ray = args.ndc_ray
    N_entropy = args.N_entropy
    aabb = train_dataset.scene_bbox.to(device)
    reso_cur = N_to_reso(args.N_voxel_init, aabb)
    nSamples = min(args.nSamples, cal_n_samples(reso_cur,args.step_ratio))    
    stop_count = 0
  
    # init resolution
    upsamp_list = args.upsamp_list
    update_AlphaMask_list = args.update_AlphaMask_list
    n_lamb_sigma = args.n_lamb_sigma
    n_lamb_sh = args.n_lamb_sh
    
    if args.add_timestamp:
        logfolder = f'{args.basedir}/{args.expname}{datetime.datetime.now().strftime("-%Y%m%d-%H%M%S")}'
    else:
        logfolder = f'{args.basedir}/{args.N_train_imgs}{args.expname}_{int(args.n_iters/1000)}k_spl{nSamples}_free({args.free_reg})_infoN({args.info_nerf})_noBatch({args.no_batching})entropy({args.entropy})smoo({args.smoothing})_v{args.view_pe}p{args.pos_pe}f{args.fea_pe}'
                     
    if args.overwrt and os.path.exists(logfolder):
      import shutil
      shutil.rmtree(logfolder)

    # init log file
    os.makedirs(logfolder, exist_ok=True)
    os.makedirs(f'{logfolder}/imgs_vis', exist_ok=True)
    os.makedirs(f'{logfolder}/imgs_rgba', exist_ok=True)
    os.makedirs(f'{logfolder}/rgba', exist_ok=True)
    os.makedirs(f'{logfolder}/plot_image', exist_ok=True)
    
    summary_writer = SummaryWriter(logfolder)

    # init parameters
    # tensorVM, renderer = init_parameters(args, train_dataset.scene_bbox.to(device), reso_list[0])

    ckpt = '{logfolder}/{args.expname}.th'
    # file_path = Path(args.ckpt)
    file_path = Path(ckpt)
    file_path.is_file()
    if file_path.is_file():
        # ckpt = torch.load(args.ckpt, map_location=device)
        ckpt = torch.load(ckpt, map_location=device)
        kwargs = ckpt['kwargs']
        kwargs.update({'device':device})
        tensorf = eval(args.model_name)(**kwargs)
        tensorf.load(ckpt)
    else:
        tensorf = eval(args.model_name)(args, aabb, reso_cur, device, density_n_comp=n_lamb_sigma, appearance_n_comp=n_lamb_sh, near_far=near_far) 

    grad_vars = tensorf.get_optparam_groups(args.lr_init, args.lr_basis)
    if args.lr_decay_iters > 0:
        lr_factor = args.lr_decay_target_ratio**(1/args.lr_decay_iters)
    else:
        args.lr_decay_iters = args.n_iters
        lr_factor = args.lr_decay_target_ratio**(1/args.n_iters)

    print("lr decay", args.lr_decay_target_ratio, args.lr_decay_iters)
    
    optimizer = torch.optim.Adam(grad_vars, betas=(0.9,0.99))

    #linear in logrithmic space
    N_voxel_list = (torch.round(torch.exp(torch.linspace(np.log(args.N_voxel_init), np.log(args.N_voxel_final), len(upsamp_list)+1))).long()).tolist()[1:]

    torch.cuda.empty_cache()
    PSNRs,PSNRs_test,losses= [0],[0],[0]

    allrays, allrgbs = train_dataset.all_rays.to(device), train_dataset.all_rgbs.to(device)

    if not args.ndc_ray:
        allrays, allrgbs = tensorf.filtering_rays(allrays, allrgbs, bbox_only=True)

    # Determine N_rand, N_entropy info nerf
    use_batching = not args.no_batching
    if args.info_nerf:
        if use_batching:
            if args.entropy and (args.N_entropy!=0):
                N_rand = args.train_batch_size - args.N_entropy
            else:
                N_rand = args.train_batch_size
        else:
            if args.entropy and (args.N_entropy!=0):
                N_rand = args.train_batch_size-args.N_entropy
                if args.smoothing:
                    N_rand = int((args.train_batch_size-2*args.N_entropy)/2)    
            else:
                N_rand = int(args.train_batch_size/2)
    else:
        N_rand = args.train_batch_size


    trainingSampler = SimpleSampler(allrays.shape[0], N_rand)
    entropySampler = SimpleSampler(allrays.shape[0], N_entropy)

    print('========================= INITIAL PARAMETERS =========================')    
    print(f"\nUpdate AlphaMask list at {update_AlphaMask_list[:2]}")

    Ortho_reg_weight = args.Ortho_weight
    print("Initial Ortho_reg_weight", Ortho_reg_weight)

    L1_reg_weight = args.L1_weight_inital
    print("Initial L1_reg_weight", L1_reg_weight)
    TV_weight_density, TV_weight_app = args.TV_weight_density, args.TV_weight_app
    tvreg = TVLoss()
    print(f"Initial TV_weight density: {TV_weight_density} appearance: {TV_weight_app}")
    print('\n\n')
  
    pbar = tqdm(range(args.n_iters), miniters=args.progress_refresh_rate, file=sys.stdout)


    from collections import defaultdict
    history = defaultdict(list)

    if args.entropy:
        N_entropy = args.N_entropy
        fun_entropy_loss = nerf_math.EntropyLoss(args, nSamples)

    if args.smoothing:
        get_near_c2w = ray_utils.GetNearC2W(args)
        fun_KL_divergence_loss = nerf_math.SmoothingLoss(args)


    for iteration in pbar:
        if args.info_nerf:
            if use_batching:
                # Random over all images
                train_ray_idx = trainingSampler.nextids()
                batch_rays, target_s = allrays[train_ray_idx].to(device), allrgbs[train_ray_idx].to(device)
                if args.entropy and (args.N_entropy!=0):
                    entropy_ray_idx = entropySampler.nextids()
                    batch_rays_entropy = allrays[entropy_ray_idx].to(device)
            else:
                train_ray_idx = trainingSampler.nextids()
                batch_rays, target_s = allrays[train_ray_idx].to(device), allrgbs[train_ray_idx].to(device)
              
                # Random from one image
                img_i = np.random.choice([0,1,2,3,4,5,6,7,8,9])
                target = stack_train_dataset.all_rgbs[img_i].to(device)
                    
                rgb_pose = stack_train_dataset.poses[img_i][:3, :4].to(device)
                W, H = stack_train_dataset.img_wh
                focal = (stack_train_dataset.focal_x, stack_train_dataset.focal_y)
                if N_rand is not None:
                    directions = ray_utils.get_ray_directions(H, W, focal).to(device)  # (H, W, 3), (H, W, 3)
                    rays_o, rays_d = ray_utils.get_rays(directions, rgb_pose)
                    rays_o = rays_o.reshape((H, W, 3)).to(device)
                    rays_d = rays_d.reshape((H, W, 3)).to(device)

                    if iteration < args.precrop_iters:
                        dH = int(H//2 * args.precrop_frac)
                        dW = int(W//2 * args.precrop_frac)
                        coords = torch.stack(
                            torch.meshgrid(
                                torch.linspace(H//2 - dH, H//2 + dH - 1, 2*dH), 
                                torch.linspace(W//2 - dW, W//2 + dW - 1, 2*dW)
                            ), -1).to(device)
                        """if i == start:
                            print(f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {args.precrop_iters}")                """
                    else:
                        coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1).to(device)  # (H, W, 2)

                    coords = torch.reshape(coords, [-1,2])  # (H * W, 2)
                    select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
                    select_coords = coords[select_inds].long()  # (N_rand, 2)
                    rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                    rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                    batch_rays = torch.cat([rays_o, rays_d], 1) # (N_rand, 6)
                    target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)

                    if args.smoothing:
                        rgb_near_pose = get_near_c2w(rgb_pose, iter_=iteration)
                        directions = ray_utils.get_ray_directions(H, W, focal).to(device)  # (H, W, 3), (H, W, 3)
                        near_rays_o, near_rays_d = ray_utils.get_rays(directions, torch.Tensor(rgb_near_pose))
                        near_rays_o = near_rays_o.reshape((H, W, 3))
                        near_rays_d = near_rays_d.reshape((H, W, 3))
                        near_rays_o = near_rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                        near_rays_d = near_rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                        near_batch_rays = torch.cat([near_rays_o, near_rays_d], 1) # (N_rand, 6)

                ########################################################
                #            Sampling for unseen rays                  #
                ########################################################
                if args.entropy and (args.N_entropy !=0):
                    img_i = np.random.choice(10)
                    target = stack_train_dataset.all_rgbs[img_i].to(device)
                    pose = stack_train_dataset.poses[img_i][:3, :4].to(device)
                    
                    if args.smooth_sampling_method == 'near_pixel':
                        if args.smooth_pixel_range is None:
                            raise Exception('The near pixel is not defined')
                        # padding=args.smooth_pixel_range
                        directions = ray_utils.get_ray_directions(H, W, focal).to(device)  # (H, W, 3), (H, W, 3)
                        rays_o, rays_d = ray_utils.get_rays(directions, pose)
                        rays_o = rays_o.reshape((H, W, 3))
                        rays_d = rays_d.reshape((H, W, 3))
                    else:
                        directions = ray_utils.get_ray_directions(H, W, focal).to(device) # (H, W, 3), (H, W, 3)
                        rays_o, rays_d = ray_utils.get_rays(directions, pose)
                        rays_o = rays_o.reshape((H, W, 3))
                        rays_d = rays_d.reshape((H, W, 3))
                    
                    if iteration < args.precrop_iters:
                        dH = int(H//2 * args.precrop_frac)
                        dW = int(W//2 * args.precrop_frac)
                        coords = torch.stack(
                            torch.meshgrid(
                                torch.linspace(H//2 - dH, H//2 + dH - 1, 2*dH), 
                                torch.linspace(W//2 - dW, W//2 + dW - 1, 2*dW)
                            ), -1)
                        """if i == start:
                            print(f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {args.precrop_iters}")   """
                    else:
                        if args.smooth_sampling_method == 'near_pixel':
                            padding = args.smooth_pixel_range
                            coords = torch.stack(
                                    torch.meshgrid(torch.linspace(padding, H-1+padding, H), 
                                    torch.linspace(padding, W-1+padding, W)), -1)  # (H, W, 2)
                        else:
                            coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)
                    
                    coords = torch.reshape(coords, [-1,2])  # (H * W, 2)
                    select_inds = np.random.choice(coords.shape[0], size=[args.N_entropy], replace=False)  # (N_rand,)
                    select_coords = coords[select_inds].long()  # (N_rand, 2)
                    rays_o_ent = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                    rays_d_ent = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                    batch_rays_entropy = torch.cat([rays_o_ent, rays_d_ent], 1) # (N_rand, 6)
                    
                    ########################################################
                    #   Ray sampling for information gain reduction loss   #
                    ########################################################

                    if args.smoothing:
                        if args.smooth_sampling_method == 'near_pixel':
                            near_select_coords = ray_utils.get_near_pixel(select_coords, args.smooth_pixel_range)
                            ent_near_rays_o = rays_o[near_select_coords[:, 0], near_select_coords[:, 1]]  # (N_rand, 3)
                            ent_near_rays_d = rays_d[near_select_coords[:, 0], near_select_coords[:, 1]]  # (N_rand, 3)
                            ent_near_batch_rays = torch.cat([ent_near_rays_o, ent_near_rays_d], 1) # (2, N_rand, 3)
                        elif args.smooth_sampling_method == 'near_pose':
                            ent_near_pose = get_near_c2w(pose,iter_=iteration)
                            directions = ray_utils.get_ray_directions(H, W, focal).to(device)  # (H, W, 3), (H, W, 3)
                            ent_near_rays_o, ent_near_rays_d = ray_utils.get_rays(directions, torch.Tensor(ent_near_pose))
                            ent_near_rays_o = ent_near_rays_o.reshape((H, W, 3))
                            ent_near_rays_d = ent_near_rays_d.reshape((H, W, 3))
                            ent_near_rays_o = ent_near_rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                            ent_near_rays_d = ent_near_rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                            ent_near_batch_rays = torch.cat([ent_near_rays_o, ent_near_rays_d], 1) # (2, N_rand, 3)
        else:
            train_ray_idx = trainingSampler.nextids()
            batch_rays, target_s = allrays[train_ray_idx].to(device), allrgbs[train_ray_idx].to(device)

        if args.entropy and (args.N_entropy !=0) and args.info_nerf:
            batch_rays = torch.cat([batch_rays, batch_rays_entropy], 0)

        if args.smoothing and not use_batching and args.info_nerf:
            if args.entropy and (args.N_entropy !=0):
                # print(batch_rays.shape, near_batch_rays.shape, ent_near_batch_rays.shape)
                batch_rays = torch.cat([batch_rays, near_batch_rays, ent_near_batch_rays], 0)
            else: 
                batch_rays = torch.cat([batch_rays, near_batch_rays], 0)

        #rgb_map, alphas_map, depth_map, weights, uncertainty
        if args.free_reg:
            rgb_map, disp_map, all_rgb_voxel, sigma, n_valib_rgb, acc_map, alpha, dists = renderer(
              batch_rays, 
              tensorf, 
              iteration, 
              total_freq_reg_step=args.freq_reg_ratio*args.n_iters,
              chunk=args.train_batch_size,
              N_samples=nSamples, 
              white_bg = white_bg, 
              ndc_ray=ndc_ray, 
              device=device, 
              is_train=True
              )
        else:
            rgb_map, disp_map, all_rgb_voxel, sigma, n_valib_rgb, acc_map, alpha, dists = renderer(
              batch_rays, 
              tensorf, 
              -1, 
              total_freq_reg_step=args.freq_reg_ratio*args.n_iters,
              chunk=args.train_batch_size,
              N_samples=nSamples, 
              white_bg = white_bg, 
              ndc_ray=ndc_ray, 
              device=device, 
              is_train=True
              )

        if n_valib_rgb != None:
          number_valib_rgb = n_valib_rgb

        if args.info_nerf:
            acc_raw = acc_map 
            alpha_raw = alpha
            dists_raw = dists

        rgb_map = rgb_map[:N_rand]        
        disp_map = disp_map[:N_rand]        
        acc_map = acc_map[:N_rand]        
        
        loss = torch.mean((rgb_map - target_s.to(device)) ** 2)   

        # loss  
        total_loss = loss
        loss = round(loss.detach().item(), 4)        

        entropy_ray_zvals_loss = 0
        smoothing_loss = 0
        loss_reg = 0
        loss_reg_L1 = 0
        occ_reg_loss = 0

        '''print(N_rand, N_entropy)
        print(rgb_map.shape, target_s.shape)
        print(alpha_raw.shape, acc_raw.shape)
        exit()'''

        if args.entropy and args.info_nerf:
            entropy_ray_zvals_loss = fun_entropy_loss.ray_zvals(alpha_raw, acc_raw)
            # history['entropy_ray_zvals'] = entropy_ray_zvals_loss.detach().numpy()
            
        if args.entropy_end_iter is not None:
            if iteration > args.entropy_end_iter:
                entropy_ray_zvals_loss = 0

        total_loss += args.entropy_ray_zvals_lambda * entropy_ray_zvals_loss                

        smoothing_lambda = args.smoothing_lambda * args.smoothing_rate ** (int(iteration/args.smoothing_step_size))
        if args.smoothing and args.info_nerf and args.no_batching:
            smoothing_loss = fun_KL_divergence_loss(alpha)
            # history['KL_loss'] = smoothing_loss.detach().numpy()
            if args.smoothing_end_iter is not None:
                if iteration > args.smoothing_end_iter:
                    smoothing_loss = 0        
        total_loss += smoothing_lambda * smoothing_loss

        if Ortho_reg_weight > 0:
            loss_reg = tensorf.vector_comp_diffs()
            total_loss += Ortho_reg_weight*loss_reg
            summary_writer.add_scalar('train/reg', loss_reg.detach().item(), global_step=iteration)
        
        if L1_reg_weight > 0:
            loss_reg_L1 = tensorf.density_L1()
            total_loss += L1_reg_weight*loss_reg_L1
            summary_writer.add_scalar('train/reg_l1', loss_reg_L1.detach().item(), global_step=iteration)

        if args.occ_reg_loss_mult > 0:
            occ_reg_loss = nerf_math.lossfun_occ_reg(
                all_rgb_voxel, 
                sigma, 
                reg_range=args.occ_reg_range,
                wb_prior=args.occ_wb_prior, 
                wb_range=args.occ_wb_range)
            occ_reg_loss = args.occ_reg_loss_mult * occ_reg_loss
            total_loss += occ_reg_loss

        if TV_weight_density > 0:
            TV_weight_density *= lr_factor
            loss_tv = tensorf.TV_loss_density(tvreg) * TV_weight_density
            total_loss = total_loss + loss_tv
            summary_writer.add_scalar('train/reg_tv_density', loss_tv.detach().item(), global_step=iteration)

        if TV_weight_app > 0:
            TV_weight_app *= lr_factor
            loss_tv = tensorf.TV_loss_app(tvreg)*TV_weight_app
            total_loss = total_loss + loss_tv
            summary_writer.add_scalar('train/reg_tv_app', loss_tv.detach().item(), global_step=iteration)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        PSNRs.append(-10.0 * np.log(loss) / np.log(10.0))
        losses.append(loss)
        summary_writer.add_scalar('train/PSNR', PSNRs[-1], global_step=iteration)
        summary_writer.add_scalar('train/mse', loss, global_step=iteration)


        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * lr_factor
            
        if iteration % args.progress_refresh_rate == 0:
            pbar.set_description(
                f'Iteration {iteration:05d}:'
                + f' valib_rgb = {number_valib_rgb[0]}/{number_valib_rgb[1]}({round((number_valib_rgb[0]*100)/number_valib_rgb[1], 2)}%)'
                + f' total_loss = {total_loss:.6f}'
                + f' train_psnr = {float(np.mean(PSNRs)):.2f}'
                + f' test_psnr = {float(np.mean(PSNRs_test)):.2f}'   
                + f' entropy = {args.entropy_ray_zvals_lambda * entropy_ray_zvals_loss:.6f}'
                + f' smoothing = {smoothing_lambda * smoothing_loss:.6f}'
                + f' ortho = {Ortho_reg_weight*loss_reg:.6f}'
                + f' L1 = {L1_reg_weight*loss_reg_L1:.6f}'
                + f' occ = {args.occ_reg_loss_mult*occ_reg_loss:.6f}'
                + f' mse = {loss:.6f}'
            )
                
            

        if iteration % args.train_vis_every == 0:
            if iteration % args.vis_every == 0:
                PSNRs_test = PSNRs_calculate(
                  test_dataset,
                  tensorf, 
                  args, 
                  renderer, 
                  iteration, 
                  args.n_iters, 
                  N_vis=-1, 
                  N_samples=nSamples,
                  white_bg=white_bg, 
                  ndc_ray=ndc_ray, 
                  device=device)
            history['iteration'].append(iteration)
            history['train_psnr'].append(round(float(np.mean(PSNRs)), 2))
            history['test_psnr'].append(round(float(np.mean(PSNRs_test)), 2))
            history['mse'].append(round(loss, 5))
            history['pc_valib_rgb'].append(round(number_valib_rgb[0]/number_valib_rgb[1], 2))        

            for param_group in tensorf.get_optparam_groups():
                history[param_group['name']].append(float(round(param_group['lr'] * lr_factor, 5)))

            save_rendered_image_per_train(
              train_dataset       = train_visual,
              test_dataset        = test_visual,
              tensorf             = tensorf, 
              renderer            = renderer,
              current_step        = iteration,
              total_train_step    = args.n_iters, 
              logs                = history,
              step_size           = args.train_vis_every,
              savePath            = f'{logfolder}/gif/',
              N_vis               = -1, 
              N_samples           = -1, 
              white_bg            = white_bg, 
              ndc_ray             = ndc_ray,
              device              = device
              )

        """# early stop
        if len(history['test_psnr']) > 10:
            if abs(history['test_psnr'][-1] - history['test_psnr'][-2]) < args.stop_thresh:
                stop_count += 1
                if stop_count == args.stop_loop:
                    break
            else:
                stop_count = 0"""

            
        if iteration in update_AlphaMask_list:

            if reso_cur[0] * reso_cur[1] * reso_cur[2]<256**3:# update volume resolution
                reso_mask = reso_cur
            new_aabb = tensorf.updateAlphaMask(tuple(reso_mask))
            
            if iteration == update_AlphaMask_list[0]:
            # if iteration in update_AlphaMask_list[:2]:
                tensorf.shrink(new_aabb)
                L1_reg_weight = args.L1_weight_rest
                print("continuing L1_reg_weight", L1_reg_weight)

            if not args.ndc_ray and iteration in update_AlphaMask_list[1:]:
            # if not args.ndc_ray and iteration in update_AlphaMask_list[:2]:
                # filter rays outside the bbox
                allrays,allrgbs = tensorf.filtering_rays(allrays,allrgbs)
                trainingSampler = SimpleSampler(allrays.shape[0], N_rand)
                entropySampler  = SimpleSampler(allrays.shape[0], N_entropy)


        if iteration in upsamp_list:
            n_voxels = N_voxel_list.pop(0)
            reso_cur = N_to_reso(n_voxels, tensorf.aabb)
            nSamples = min(args.nSamples, cal_n_samples(reso_cur,args.step_ratio))
            tensorf.upsample_volume_grid(reso_cur)

            if args.lr_upsample_reset:
                print("reset lr to initial")
                lr_scale = 1 #0.1 ** (iteration / args.n_iters)
            else:
                lr_scale = args.lr_decay_target_ratio ** (iteration / args.n_iters)
            grad_vars = tensorf.get_optparam_groups(args.lr_init*lr_scale, args.lr_basis*lr_scale)
            optimizer = torch.optim.Adam(grad_vars, betas=(0.9, 0.99))
        
    tensorf.save(f'{logfolder}/{args.expname}.th')


    for key in list(history.keys())[1:]:
        plt.figure()
        plt.plot(history['iteration'], history[key], linestyle='-', color='b', label='Data Points')
        plt.xlabel('Iteration')
        plt.ylabel(f'{key.capitalize()}')
        plt.title(f'{key.capitalize()} Plot')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{logfolder}/plot_image/{key}_plot.png')
        plt.close()
    np.savez(f'{logfolder}/{args.expname}_history.npz', **history)

    if args.render_train:
        os.makedirs(f'/content/metric_plot/0525/imgs_traning', exist_ok=True)
        
        train_dataset = dataset('/content/drive/MyDrive/Dataset/THuman2.0/rendered_images/training_render', 
        split='train', downsample=args.downsample_train, is_stack=True)

        PSNR_test = evaluation(
          train_dataset,
          tensorf, 
          args, 
          renderer, 
          iteration,
          args.freq_reg_ratio*args.n_iters, 
          f'{logfolder}/imgs_train_all/',
          N_vis=-1, 
          N_samples=-1, 
          white_bg = white_bg, 
          ndc_ray=ndc_ray,
          device=device
          )
        print(f'======> {args.expname} test all psnr: {np.mean(PSNR_test)} <========================')


    if args.render_test:
        os.makedirs(f'{logfolder}/imgs_test_all', exist_ok=True)
        PSNR_test = evaluation(
          test_dataset,
          tensorf, 
          args, 
          renderer, 
          iteration, 
          args.freq_reg_ratio*args.n_iters, 
          f'{logfolder}/imgs_test_all/',
          N_vis=-1, 
          N_samples=-1, 
          white_bg = white_bg, 
          ndc_ray=ndc_ray,
          device=device
          )
        summary_writer.add_scalar('test/psnr_all', np.mean(PSNR_test), global_step=iteration)
        print(f'======> {args.expname} test all psnr: {np.mean(PSNRs_test)} <========================')


    if args.render_path:
        c2ws = test_dataset.render_path
        # c2ws = test_dataset.poses
        print('========>',c2ws.shape)
        os.makedirs(f'{logfolder}/imgs_path_all', exist_ok=True)
        evaluation_path(
          test_dataset,
          tensorf, 
          c2ws, 
          renderer, 
          iteration, 
          args.freq_reg_ratio*args.n_iters, 
          f'{logfolder}/imgs_path_all/',
          N_vis=-1, 
          N_samples=-1, 
          white_bg = white_bg, 
          ndc_ray=ndc_ray,
          device=device
          )

    create_gif(f"{logfolder}/gif/plot/{args.train_vis_every}", f"{logfolder}/gif/training.gif")

    return f'{logfolder}/{args.expname}.th'


if __name__ == '__main__':

    torch.set_default_dtype(torch.float32)
    torch.manual_seed(20211202)
    np.random.seed(20211202)

    args = config_parser()

    if  args.export_mesh and args.config:
        ckpt_path = reconstruction(args)        
        export_mesh(args, ckpt_path)  

        import shutil 
        shutil.copy(args.config, ckpt_path[:-3]) 

    elif args.export_mesh:
        export_mesh(args)        

    if args.render_only and (args.render_test or args.render_path):
        print(render_test(args))