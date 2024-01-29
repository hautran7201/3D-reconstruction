import datetime
import json
import os
import random
import sys
import time
import torch
import numpy as np
import shutil
from torch._C import _is_tracing

from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from dataLoader import dataset_dict
from opt import config_parser
from renderer import *
from utils import *
from pathlib import Path
from collections import defaultdict

from models.learned_regularisation.intrinsics import Intrinsics
from models.learned_regularisation.patch_pose_generator import PatchPoseGenerator, FrustumChecker, FrustumRegulariser
from models.learned_regularisation.patch_regulariser import load_patch_diffusion_model, \
    PatchRegulariser, LLFF_DEFAULT_PSEUDO_INTRINSICS, HUMAN_DEFAULT_PSEUDO_INTRINSICS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SimpleSampler:
    def __init__(self, total, batch):
        self.total = total
        self.batch = batch
        self.curr = total
        self.ids = None

    def nextids(self):
        self.curr += self.batch
        if self.curr + self.batch > self.total:
            self.ids = torch.LongTensor(np.random.permutation(self.total))
            self.curr = 0
        return self.ids[self.curr : self.curr + self.batch]


@torch.no_grad()
@torch.no_grad()
def export_mesh(args, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=device)
    kwargs = ckpt['kwargs']
    kwargs.update({'device': device})
    tensorf = eval(args.model_name)(args, **kwargs)
    tensorf.load(ckpt)

    alpha,_ = tensorf.getDenseAlpha()
    convert_sdf_samples_to_ply(
      alpha.cpu(), 
      f'{ckpt_path[:-3]}.ply',
      bbox=tensorf.aabb.cpu(), 
      level=0.005
    )


@torch.no_grad()
def render_test(args):
    # init dataset
    dataset = dataset_dict[args.dataset_name]
    test_dataset = dataset(
        args.datadir,
        split="test",
        downsample=args.downsample_train,
        is_stack=True,
    )
    white_bg = test_dataset.white_bg
    ndc_ray = args.ndc_ray

    if not os.path.exists(args.ckpt):
        print("the ckpt path does not exists!!")
        return

    ckpt = torch.load(args.ckpt, map_location=device)
    kwargs = ckpt["kwargs"]
    kwargs.update({"device": device})
    occ_grid = None
    if args.occ_grid_reso > 0:
        occ_grid = nerfacc.OccGridEstimator(
            roi_aabb=ckpt["state_dict"]["occGrid.aabbs"][0],
            resolution=args.occ_grid_reso,
        ).to(device)
    tensorf = eval(args.model_name)(args, **kwargs)

    tensorf.load(ckpt)

    logfolder = os.path.dirname(args.ckpt)
    if args.render_train:
        os.makedirs(f"{logfolder}/imgs_train_all", exist_ok=True)
        train_dataset = dataset(
            args.datadir,
            split="train",
            downsample=args.downsample_train,
            is_stack=True,
        )
        PSNRs_test = evaluation(
            train_dataset,
            tensorf,
            args,
            renderer,
            f"{logfolder}/imgs_train_all/",
            N_vis=-1,
            N_samples=-1,
            white_bg=white_bg,
            ndc_ray=ndc_ray,
            device=device,
        )
        print(
            f"======> {args.expname} train all psnr: {np.mean(PSNRs_test)} <========================"
        )

    if args.render_test:
        os.makedirs(f"{logfolder}/imgs_test_all", exist_ok=True)
        PSNRs_test = evaluation(
            test_dataset,
            tensorf,
            args,
            renderer,
            f"{logfolder}/imgs_test_all/",
            N_vis=-1,
            N_samples=-1,
            white_bg=white_bg,
            ndc_ray=ndc_ray,
            device=device,
        )
        print(
            f"======> {args.expname} test all psnr: {np.mean(PSNRs_test)} <========================"
        )

    if args.render_path:
        c2ws = test_dataset.render_path
        os.makedirs(f"{logfolder}/imgs_path_all", exist_ok=True)
        print(f"{logfolder}/imgs_path_all")
        evaluation_path(
            test_dataset,
            tensorf,
            c2ws,
            renderer,
            f"{logfolder}/imgs_path_all/",
            N_vis=-1,
            N_samples=-1,
            white_bg=white_bg,
            ndc_ray=ndc_ray,
            device=device,
        )


def reconstruction(args):
    # init dataset

    dataset = dataset_dict[args.dataset_name]
    if len(args.train_idxs) == 0:
        train_dataset = dataset(
          args.datadir, 
          split='train', 
          downsample=args.downsample_train, 
          is_stack=False, 
          N_imgs=args.N_train_imgs
        )
    else:
        train_dataset = dataset(
          args.datadir, 
          split='train', 
          downsample=args.downsample_train, 
          is_stack=False, 
          indexs=args.train_idxs
        )

    if len(args.val_idxs) == 0:
        test_dataset = dataset(
          args.datadir, 
          split='test', 
          downsample=args.downsample_train, 
          is_stack=True, 
          N_imgs=args.N_test_imgs
        )
    else:
        test_dataset = dataset(
          args.datadir, 
          split='test', 
          downsample=args.downsample_train, 
          is_stack=True, 
          indexs=args.val_idxs
        )

    if len(args.test_idxs) == 0:
        final_test_dataset = dataset(
          args.datadir, 
          split='test', 
          downsample=args.downsample_train, 
          is_stack=True, 
          N_imgs=args.N_test_imgs
        )
    else:
        final_test_dataset = dataset(
          args.datadir, 
          split='test', 
          downsample=args.downsample_train, 
          is_stack=True, 
          indexs=args.test_idxs
        )

    # Observation
    train_visual = dataset(
      args.datadir, 
      split='train', 
      downsample=args.downsample_train, 
      is_stack=True, 
      tqdm=False, 
      indexs=[24]
    )

    test_visual = dataset(
      args.datadir, 
      split='test', 
      downsample=args.downsample_train, 
      is_stack=True, 
      tqdm=False, 
      indexs=[24]
    )

    white_bg = train_dataset.white_bg
    near_far = train_dataset.near_far
    ndc_ray = args.ndc_ray

    # init resolution
    upsamp_list = args.upsamp_list
    update_AlphaMask_list = args.update_AlphaMask_list
    n_lamb_sigma = args.n_lamb_sigma
    n_lamb_sh = args.n_lamb_sh
    mask_ratio_list = args.mask_ratio_list


    if args.add_timestamp:
        logfolder = f'{args.basedir}/{args.expname}{datetime.datetime.now().strftime("-%Y%m%d-%H%M%S")}'
    else:
        logfolder = f"{args.basedir}/{args.expname}"

    if args.overwrt and os.path.exists(logfolder):
        shutil.rmtree(logfolder)


    # init log file
    os.makedirs(logfolder, exist_ok=True)
    os.makedirs(f"{logfolder}/imgs_vis", exist_ok=True)
    os.makedirs(f"{logfolder}/imgs_rgba", exist_ok=True)
    os.makedirs(f"{logfolder}/rgba", exist_ok=True)
    summary_writer = SummaryWriter(logfolder)

    # init parameters
    # tensorVM, renderer = init_parameters(args, train_dataset.scene_bbox.to(device), reso_list[0])
    aabb = train_dataset.scene_bbox.to(device)
    reso_cur = N_to_reso(args.N_voxel_init, aabb)
    nSamples = min(args.nSamples, cal_n_samples(reso_cur, args.step_ratio))

    occ_grid = None
    if args.occ_grid_reso > 0:
        import nerfacc
        occ_grid = nerfacc.OccGridEstimator(roi_aabb=aabb.reshape(-1),resolution=args.occ_grid_reso).to(device)

    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt, map_location=device)
        kwargs = ckpt["kwargs"]
        kwargs.update({"device": device})
        tensorf = eval(args.model_name)(**kwargs, occGrid=occ_grid)
        tensorf.load(ckpt)
    else:
        tensorf = eval(args.model_name)(
            args,
            aabb,
            reso_cur,
            device,
            density_n_comp=n_lamb_sigma,
            appearance_n_comp=n_lamb_sh,
            app_dim=args.data_dim_color,
            near_far=near_far,
            shadingMode=args.shadingMode,
            occGrid=occ_grid,
            alphaMask_thres=args.alpha_mask_thre,
            density_shift=args.density_shift,
            distance_scale=args.distance_scale,
            pos_pe=args.pos_pe,
            view_pe=args.view_pe,
            fea_pe=args.fea_pe,
            featureC=args.featureC,
            step_ratio=args.step_ratio,
            fea2denseAct=args.fea2denseAct,
        )


    # Don't update aabb and invaabbSize for occ.
    def occ_normalize_coord(xyz_sampled):
        return (xyz_sampled - tensorf.aabb[0]) * tensorf.invaabbSize - 1


    # Set optimizers
    grad_vars = tensorf.get_optparam_groups(args.lr_init, args.lr_basis)
    optimizer = torch.optim.Adam(grad_vars, betas=(0.9, 0.99))
    
    if args.lr_decay_iters > 0:
        lr_factor = args.lr_decay_target_ratio ** (1 / args.lr_decay_iters)
    else:
        args.lr_decay_iters = args.n_iters
        lr_factor = args.lr_decay_target_ratio ** (1 / args.n_iters)

    print("lr decay", args.lr_decay_target_ratio, args.lr_decay_iters)


    # linear in logrithmic space
    N_voxel_list = (
        torch.round(
            torch.exp(
                torch.linspace(
                    np.log(args.N_voxel_init),
                    np.log(args.N_voxel_final),
                    len(upsamp_list) + 1,
                )
            )
        ).long()
    ).tolist()[1:]


    # Set log
    torch.cuda.empty_cache()
    PSNRs, PSNRs_train, PSNRs_test, history = [], [0], [0], defaultdict(list)
    

    # Get train data
    allrays, allrgbs, allmasks = train_dataset.all_rays, train_dataset.all_rgbs, train_dataset.all_masks

    if not args.ndc_ray:
        allrays, allrgbs = tensorf.filtering_rays(allrays, allrgbs, bbox_only=True)
    trainingSampler = SimpleSampler(allrays.shape[0], args.batch_size)


    # Set loss weight
    Ortho_reg_weight = args.Ortho_weight
    L1_reg_weight = args.L1_weight_inital
    TV_weight_density, TV_weight_app = (args.TV_weight_density,args.TV_weight_app)
    tvreg = TVLoss()
    print("initial Ortho_reg_weight", Ortho_reg_weight)
    print("initial L1_reg_weight", L1_reg_weight)
    print(f"initial TV_weight density: {TV_weight_density} appearance: {TV_weight_app}")


    # run_tic = time.time()
    # Load diffusion model
    patch_regulariser_path = 'models/diffusion_checkpoint/rgbd-patch-diffusion.pt'
    patch_diffusion_model = load_patch_diffusion_model(Path(patch_regulariser_path), device)

    fx, fy, cx, cy = train_dataset.intrinsics_info
    intrinsics = Intrinsics(
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        width=train_dataset.w,
        height=train_dataset.h,
    )

    frustum_checker = FrustumChecker(fov_x_rads=train_dataset.fov_x, fov_y_rads=train_dataset.fov_y)
    frustum_regulariser = FrustumRegulariser(
        intrinsics=intrinsics,
        reg_strength=1e-5,  # NB in the trainer this gets multiplied by the strength params passed in via the args
        min_near=train_dataset.near_far[0],
        poses=[torch.Tensor(pose).to(device) for pose in train_dataset.poses],
    )

    pose_generator = PatchPoseGenerator(
                poses=torch.Tensor(train_dataset.scale_poses),
                spatial_perturbation_magnitude=0.2,
                angular_perturbation_magnitude_rads=0.2 * np.pi,
                no_perturb_prob=0.,
                frustum_checker=frustum_checker if args.frustum_check_patches else None
    )

    pseudo_intrinsics = HUMAN_DEFAULT_PSEUDO_INTRINSICS       

    if args.diffu_reg:
        patch_regulariser = PatchRegulariser(
            pose_generator = pose_generator,
            patch_diffusion_model = patch_diffusion_model,
            full_image_intrinsics = pseudo_intrinsics,
            device = device,
            planar_depths = True,
            patch_size = 48,
            frustum_regulariser = frustum_regulariser if args.frustum_regularise_patches else None,
            sample_downscale_factor = args.patch_sample_downscale_factor,
            uniform_in_depth_space = args.normalise_diffusion_losses
        ) 
    else:
        patch_regulariser = None

    renderer = OctreeRender_trilinear_fast(
                tensorf = tensorf,
                chunk = args.batch_size,
                ndc_ray = ndc_ray,
                white_bg = white_bg,
                device = device
    )

    pbar = tqdm(range(args.n_iters),miniters=args.progress_refresh_rate,file=sys.stdout)
    for iteration in pbar:
        ray_idx = trainingSampler.nextids()
        rays_train, rgb_train, all_mask = allrays[ray_idx], allrgbs[ray_idx].to(device), allmasks[ray_idx].to(device)

        ratio = mask_ratio_list[0]
        if args.free_reg:
            free_maskes = get_free_mask(
              pos_bl              = tensorf.pos_bit_length, 
              view_bl             = tensorf.view_bit_length, 
              fea_bl              = tensorf.fea_bit_length, 
              den_bl              = tensorf.density_bit_length,
              app_bl              = tensorf.app_bit_length,
              using_decomp_mask   = args.free_decomp,
              step                = iteration, 
              total_step          = args.n_iters * args.freq_reg_ratio_iteration, 
              ratio               = ratio,
              device              = device
            )
        else:
            free_maskes = get_free_mask()

        if tensorf.occGrid is not None:

            def occ_eval_fn(x):
                step_size = tensorf.stepSize
                # compute occupancy
                density = (
                    tensorf.feature2density(
                        tensorf.compute_densityfeature(
                            occ_normalize_coord(x), 
                            free_maskes['decomp']['den']
                        )[:, None]) * tensorf.distance_scale
                )
                return density * step_size

            tensorf.occGrid.update_every_n_steps(step=iteration, occ_eval_fn=occ_eval_fn)

        # rgb_map, alphas_map, depth_map, weights, uncertainty
        renderer.set_N_samples = nSamples
        renderer.set_mask = free_maskes
        renderer.set_is_train = True

        # output key: [rgb_map, alphas_map, depth_map, weights, uncertainty, num_samples]
        output = renderer.render(rays_train)

        # loss = torch.mean(((output['rgb'] - rgb_train) ** 2) * all_mask)
        loss = torch.mean((output['rgb_map'] - rgb_train) ** 2)

        # loss
        total_loss = loss
        """if args.occ_reg_loss_mult > 0:
            occ_reg_loss = lossfun_occ_reg(
                all_rgb_voxel, 
                sigma, 
                reg_range=args.occ_reg_range,
                wb_prior=args.occ_wb_prior, 
                wb_range=args.occ_wb_range)
            occ_reg_loss = args.occ_reg_loss_mult * occ_reg_loss
            total_loss += occ_reg_loss"""
        
        if Ortho_reg_weight > 0 and 'VM' in args.model_name:
            loss_reg = tensorf.vector_comp_diffs()
            total_loss += Ortho_reg_weight * loss_reg
            summary_writer.add_scalar(
                "train/reg", loss_reg.detach().item(), global_step=iteration
            )
        if L1_reg_weight > 0:
            loss_reg_L1 = tensorf.density_L1()
            total_loss += L1_reg_weight * loss_reg_L1
            summary_writer.add_scalar(
                "train/reg_l1",
                loss_reg_L1.detach().item(),
                global_step=iteration,
            )

        if TV_weight_density > 0:
            TV_weight_density *= lr_factor
            loss_tv = tensorf.TV_loss_density(tvreg) * TV_weight_density
            total_loss = total_loss + loss_tv
            summary_writer.add_scalar(
                "train/reg_tv_density",
                loss_tv.detach().item(),
                global_step=iteration,
            )
        if TV_weight_app > 0:
            TV_weight_app *= lr_factor
            loss_tv = tensorf.TV_loss_app(tvreg) * TV_weight_app
            total_loss = total_loss + loss_tv
            summary_writer.add_scalar(
                "train/reg_tv_app",
                loss_tv.detach().item(),
                global_step=iteration,
            )

        path_loss = 0
        if patch_regulariser is not None:
            # data
            data = train_dataset.get_img() # .get_img(30)

            # t schedule
            initial_diffusion_time = args.initial_diffusion_time
            patch_reg_start_step = args.patch_reg_start_step
            patch_reg_finish_step = args.patch_reg_finish_step
            weight_start = args.patch_weight_start
            weight_finish = args.patch_weight_finish

            global_step = iteration
            lambda_t = (global_step - patch_reg_start_step) / (patch_reg_finish_step - patch_reg_start_step)
            lambda_t = np.clip(lambda_t, 0., 1.)
            weight = weight_start + (weight_finish - weight_start) * lambda_t

            global_step = iteration
            if global_step > patch_reg_start_step:
                if global_step > patch_reg_finish_step:
                    time = 0.
                elif global_step > patch_reg_start_step:
                    time = initial_diffusion_time * (1. - lambda_t)
                else:
                    raise RuntimeError('Internal error')
                    
                p_sample_patch = 0.25
                """patch_outputs = patch_regulariser.get_diffusion_loss_with_rendered_patch(
                    model=renderer,
                    time=time
                )
                patch_outputs = patch_regulariser.get_diffusion_loss_with_sampled_patch(
                    model=renderer, time=time, image=data['image'], image_intrinsics=data['intrinsic'],
                    pose=data['pose'], iteration=-1, 
                    train_every=50 # args.train_vis_every
                )
                exit()"""
                if random.random() >= p_sample_patch: # random.random()                
                    # print('get_diffusion_loss_with_rendered_patch')
                    patch_outputs = patch_regulariser.get_diffusion_loss_with_rendered_patch(
                        model=renderer,
                        time=time
                    )
                else:
                    # print('get_diffusion_loss_with_sampled_patch')
                    patch_outputs = patch_regulariser.get_diffusion_loss_with_sampled_patch(
                        model=renderer, time=time, image=data['image'], image_intrinsics=data['intrinsic'],
                        pose=data['pose'], iteration=-1, 
                        train_every=50 # args.train_vis_every
                    )
                path_loss = weight * patch_outputs.loss
                loss += path_loss

                """# Geometric reg
                spread_loss_strength = 1e-5
                dynamic_reg_start_step  = 2000
                dynamic_reg_max_strength_step = 8000
                def get_linear_dynamic_reg_modifier():
                    # Linear scheme
                    if global_step > dynamic_reg_max_strength_step:
                        dynamic_reg_modifier = 1.
                    elif global_step > dynamic_reg_start_step:
                        dynamic_reg_modifier = (global_step - dynamic_reg_start_step) / (
                                dynamic_reg_max_strength_step - dynamic_reg_start_step)
                    else:
                        dynamic_reg_modifier = 0.
                    return dynamic_reg_modifier

                dynamic_reg_modifier = get_linear_dynamic_reg_modifier()

                spread_loss_weight = spread_loss_strength * dynamic_reg_modifier
                if True: # self.opt.apply_geom_reg_to_patches
                    loss += spread_loss_weight * patch_outputs.render_outputs['loss_dist']

                # Frustum reg
                def get_frustum_reg_str():
                    # Piecewise schedule
                    frustum_reg_initial_weight = args.frustum_reg_initial_weight # 1
                    frustum_reg_final_weight = args.frustum_reg_final_weight # 1e-2
                    frustum_reg_weight = frustum_reg_initial_weight if global_step < 100 \
                        else frustum_reg_final_weight

                    return frustum_reg_weight
                if patch_regulariser.frustum_regulariser is not None:
                    xyzs_flat = patch_outputs.render_outputs['rgb'].reshape(-1, 3)
                    weights_flat = patch_outputs.render_outputs['weight'].reshape(-1)

                    patch_frustum_reg_weight = get_frustum_reg_str()
                    patch_frustum_loss = patch_frustum_reg_weight * patch_regulariser.frustum_regulariser(
                        xyzs=xyzs_flat, weights=weights_flat, frustum_count_thresh=1,
                    )
                    loss += patch_frustum_loss"""

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        loss = loss.detach().item()

        if loss < 0:
            loss = -loss
            
        PSNRs.append(-10.0 * np.log(loss) / np.log(10.0))
        summary_writer.add_scalar(
            "train/PSNR", PSNRs[-1], global_step=iteration
        )
        summary_writer.add_scalar("train/mse", loss, global_step=iteration)

        for param_group in optimizer.param_groups:
            param_group["lr"] = param_group["lr"] * lr_factor

        # Print the current values of the losses.
        if iteration % args.progress_refresh_rate == 0:
            # step_toc = time.time()
            # + f" elapsed_time = {step_toc - run_tic:.2f}"
            pbar.set_description(
                f"Iteration {iteration:05d}:"
                + f" train_psnr = {float(np.mean(PSNRs)):.2f}"
                + f" test_psnr = {float(np.mean(PSNRs_test)):.2f}"
                + f" patch_loss = {path_loss:.6f}"
                + f" mse = {loss:.6f}"
            )
            PSNRs_train.append(float(np.mean(PSNRs)))            
            PSNRs = []
            num_rays_per_sec = 0
            num_samples_per_sec = 0
            # step_tic = time.time()
        

        if iteration % args.train_vis_every == 0:
            renderer.set_is_train = False
            if iteration % args.vis_every == 0:    
                PSNRs_test = PSNRs_calculate(
                  args,
                  tensorf,
                  test_dataset,
                  renderer, 
                  chunk=args.batch_size,
                  N_samples=nSamples,
                  white_bg=white_bg, 
                  ndc_ray=ndc_ray,                   
                  device=device)
            history['iteration'].append(iteration)
            history['train_psnr'].append(round(float(np.mean(PSNRs_train)), 2))
            history['test_psnr'].append(round(float(np.mean(PSNRs_test)), 2))
            history['mse'].append(round(loss, 5))
            # history['pc_valib_rgb'].append(round(number_valib_rgb[0]/number_valib_rgb[1], 2))        

            save_rendered_image_per_train(
              train_dataset       = train_visual,
              test_dataset        = test_visual,
              tensorf             = tensorf, 
              renderer            = renderer,
              step                = iteration,
              logs                = history,
              savePath            = f'{logfolder}/gif/',
              chunk               = args.batch_size,
              N_samples           = nSamples, 
              white_bg            = white_bg, 
              ndc_ray             = ndc_ray,
              device              = device
              )

            PSNRs_train.append(float(np.mean(PSNRs)))
            PSNRs_train = []

        if iteration in update_AlphaMask_list:
            if (
                reso_cur[0] * reso_cur[1] * reso_cur[2] < 256**3
            ):  # update volume resolution
                reso_mask = reso_cur
            new_aabb = tensorf.updateAlphaMask(tuple(reso_mask))
            if iteration == update_AlphaMask_list[0]:
                tensorf.shrink(new_aabb)
                # tensorVM.alphaMask = None
                L1_reg_weight = args.L1_weight_rest
                print("continuing L1_reg_weight", L1_reg_weight)

            if not args.ndc_ray and iteration == update_AlphaMask_list[1]:
                # filter rays outside the bbox
                allrays, allrgbs = tensorf.filtering_rays(allrays, allrgbs)
                trainingSampler = SimpleSampler(
                    allrgbs.shape[0], args.batch_size
                )

        if iteration in upsamp_list:

            if len(upsamp_list) == len(mask_ratio_list):
                ratio = mask_ratio_list[upsamp_list.index(iteration)]

            n_voxels = N_voxel_list.pop(0)
            reso_cur = N_to_reso(n_voxels, tensorf.aabb)
            nSamples = min(
                args.nSamples, cal_n_samples(reso_cur, args.step_ratio)
            )            
            tensorf.upsample_volume_grid(reso_cur)

            if args.lr_upsample_reset:
                print("reset lr to initial")
                lr_scale = 1  # 0.1 ** (iteration / args.n_iters)
            else:
                lr_scale = args.lr_decay_target_ratio ** (
                    iteration / args.n_iters
                )

            grad_vars = tensorf.get_optparam_groups(
                args.lr_init * lr_scale, args.lr_basis * lr_scale
            )
            optimizer = torch.optim.Adam(grad_vars, betas=(0.9, 0.99))

        if iteration in args.save_ckpt_every:
            tensorf.save(f'{logfolder}/{iteration//1000}k_{args.expname}.th')        
    

    tensorf.save(f'{logfolder}/final_{args.expname}.th')
    # elapsed_time = time.time() - run_tic
    # print(f"Total time {elapsed_time:.2f}s.")
    # np.savetxt(f"{logfolder}/training_time.txt", np.asarray([elapsed_time]))

    if args.render_train:
        os.makedirs(f"{logfolder}/imgs_train_all", exist_ok=True)
        renderer.set_is_train = False
        train_dataset = dataset(
            args.datadir,
            split="train",
            downsample=args.downsample_train,
            is_stack=True,
        )
        PSNRs_test = evaluation(
            train_dataset,
            tensorf,
            args,
            renderer,
            f"{logfolder}/imgs_train_all/",
            N_vis=-1,
            N_samples=-1,
            white_bg=white_bg,
            ndc_ray=ndc_ray,
            device=device,
        )
        print(
            f"======> {args.expname} test all psnr: {np.mean(PSNRs_test)} <========================"
        )

    if args.render_test:
        os.makedirs(f"{logfolder}/imgs_test_all", exist_ok=True)
        renderer.set_is_train = False
        PSNRs_test = evaluation(
            test_dataset,
            tensorf,
            args,
            renderer,
            f"{logfolder}/imgs_test_all/",
            N_vis=-1,
            N_samples=-1,
            white_bg=white_bg,
            ndc_ray=ndc_ray,
            device=device,
        )
        summary_writer.add_scalar(
            "test/psnr_all", np.mean(PSNRs_test), global_step=iteration
        )
        print(
            f"======> {args.expname} test all psnr: {np.mean(PSNRs_test)} <========================"
        )

    if args.render_path:
        c2ws = test_dataset.render_path
        # c2ws = test_dataset.poses
        print("========>", c2ws.shape)
        os.makedirs(f"{logfolder}/imgs_path_all", exist_ok=True)
        renderer.set_is_train = False
        evaluation_path(
            test_dataset,
            tensorf,
            c2ws,
            renderer,
            f"{logfolder}/imgs_path_all/",
            N_vis=-1,
            N_samples=-1,
            white_bg=white_bg,
            ndc_ray=ndc_ray,
            device=device,
        )

    np.savez(f"{logfolder}/history.npz", **history)

    create_gif(f"{logfolder}/gif/plot/vis_every", f"{logfolder}/gif/training.gif")

    return f'{logfolder}/final_{args.expname}.th'


if __name__ == '__main__':

    torch.set_default_dtype(torch.float32)
    torch.manual_seed(20211202)
    np.random.seed(20211202)

    args = config_parser()

    if args.export_mesh:
        export_mesh(args, args.ckpt)

    if args.render_only and (args.render_test or args.render_path or args.render_train):
        render_test(args)
    elif args.config:
        ckpt_path = reconstruction(args)        
        export_mesh(args, ckpt_path)  

        import shutil 
        shutil.copy(args.config, ckpt_path[:-3]+'.txt')