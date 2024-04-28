import json
import os
import random
import sys
import time
import torch
import numpy as np
import hydra
import shutil
import pytz
from datetime import datetime
from omegaconf import DictConfig, OmegaConf
from collections import defaultdict


from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from opt import config_parser
from models.tensoRF import TensorVMSplit, TensorCP
from dataLoader import dataset_dict
from renderer import (
    OctreeRender_trilinear_fast,
    evaluation, 
    create_gif,
    evaluation_path,
    save_rendered_image_per_train
)
from loss import (
    TVLoss, 
    PSNRs_calculate
)
from utils import (
    convert_sdf_samples_to_ply, 
    N_to_reso,
    cal_n_samples,
    get_free_mask,
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
renderer = OctreeRender_trilinear_fast


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
    tensorf = eval(args.model_name)(**kwargs, occGrid=occ_grid)

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


@hydra.main(version_base=None)
def reconstruction(args: DictConfig):
    # ==> Dataset
    # ================================
    dataset             = dataset_dict[args.dataset_name]
    train_dataset       = dataset(args.datadir, split='train', downsample=args.downsample_train, num_images=OmegaConf.to_object(args.train_images))
    test_dataset        = dataset(args.datadir, split='test', downsample=args.downsample_train, num_images=OmegaConf.to_object(args.test_images), is_stack=True)
    # final_test_dataset  = dataset(args.datadir, split='test', downsample=args.downsample_train, is_stack=True, num_images=args.N_train_imgs)    
    train_gift_data     = dataset(args.datadir, split='train', downsample=args.downsample_train, num_images=[26], is_stack=True)
    test_gift_data      = dataset(args.datadir, split='test', downsample=args.downsample_train, num_images=[26], is_stack=True)
    white_bg = train_dataset.white_bg
    near_far = train_dataset.near_far
    ndc_ray = args.ndc_ray


    # ==> Init resolution
    # ================================
    upsamp_list = args.upsamp_list
    update_AlphaMask_list = args.update_AlphaMask_list
    n_lamb_sigma = args.n_lamb_sigma
    n_lamb_sh = args.n_lamb_sh

    
    # ==> Init log folder
    # ================================
    timezone = pytz.timezone('Asia/Ho_Chi_Minh')
    current_time = datetime.now(timezone)
    logfolder = f'{args.basedir}/{current_time.strftime("%Y-%m-%d")}/{args.expname}'
    if args.overwrt and os.path.exists(logfolder): shutil.rmtree(logfolder)    
    os.makedirs(logfolder, exist_ok=True)
    os.makedirs(f"{logfolder}/imgs_vis", exist_ok=True)
    os.makedirs(f"{logfolder}/imgs_rgba", exist_ok=True)
    os.makedirs(f"{logfolder}/rgba", exist_ok=True)
    summary_writer = SummaryWriter(logfolder)


    # ==> Init parameters
    # ================================
    aabb = train_dataset.scene_bbox.to(device)
    gridSize = N_to_reso(args.N_voxel_init, aabb)
    nSamples = min(1e6, cal_n_samples(gridSize, args.step_ratio))
    N_voxel_list = (
        torch.round(
            torch.exp(
                torch.linspace(np.log(args.N_voxel_init), np.log(args.N_voxel_final), len(upsamp_list) + 1)
            )
        ).long()
    ).tolist()[1:]


    # ==> Load checkpoint
    # ================================
    if args.ckpt_path is not None:
        ckpt = torch.load(args.ckpt_path, map_location=device)
        kwargs = ckpt["kwargs"]
        kwargs.update({"device": device})
        tensorf = eval(args.model_name)(**kwargs)
        tensorf.load(ckpt)
    else:
        tensorf = eval(args.model_name)(
            args={
                'step_ratio': args.step_ratio,
                'fea2denseAct': args.fea2denseAct,
                'density_n_comp': args.n_lamb_sigma,
                'app_n_comp': args.n_lamb_sh,
                'app_dim': args.data_dim_color,
                'density_shift': args.density_shift,
                'distance_scale': args.distance_scale,
                'alphaMask_thres': args.alphaMask_thres,
                'shadingMode': args.shadingMode,
                'pos_pe': args.pos_pe,
                'view_pe': args.view_pe,
                'fea_pe': args.fea_pe,
                'featureC': args.featureC
            },
            aabb=aabb,
            gridSize=gridSize,
            device=device,
            near_far=train_dataset.near_far
        )

    torch.cuda.empty_cache()
    print(f"initial TV_weight density: {args.TV_weight_density} appearance: {args.TV_weight_app}")


    # ==> Loss init 
    # ================================
    Ortho_reg_weight = args.Ortho_weight
    print("initial Ortho_reg_weight", Ortho_reg_weight)
    L1_reg_weight = args.L1_weight_inital
    print("initial L1_reg_weight", L1_reg_weight)
    TV_weight_density, TV_weight_app = args.TV_weight_density, args.TV_weight_app
    tvreg = TVLoss()


    # ==> Optimzier and Learning rate
    # ================================
    if args.lr_decay_iters > 0:
        lr_factor = args.lr_decay_target_ratio ** (1 / args.lr_decay_iters)
    else:
        lr_decay_iters = args.n_iters
        lr_factor = args.lr_decay_target_ratio ** (1 / args.n_iters)
    print("lr decay", args.lr_decay_target_ratio, lr_decay_iters)

    grad_vars = tensorf.get_optparam_groups(args.lr_init, args.lr_basis)
    optimizer = torch.optim.Adam(grad_vars, betas=(0.9, 0.99))


    # ==> Training Utils
    # ================================
    PSNRs, PSNRs_train, PSNRs_test = [],[0],[0]
    history = defaultdict(list)
    run_tic = time.time()
    pbar = tqdm(
        range(args.n_iters),
        miniters=args.progress_refresh_rate,
        file=sys.stdout,
    )


    # ==> Data preparation
    # ================================
    allrays, allrgbs = train_dataset.all_rays, train_dataset.all_rgbs
    if not ndc_ray: allrays, allrgbs = tensorf.filtering_rays(allrays, allrgbs, bbox_only=True)
    trainingSampler = SimpleSampler(allrays.shape[0], args.batch_size)
    

    # Start training
    for iteration in pbar:
        ray_idx = trainingSampler.nextids()
        rays_train, rgb_train = allrays[ray_idx], allrgbs[ray_idx].to(device)


        # ==> Get mask
        # ================================
        ratio = args.mask_ratio_list[0]
        if args.free_reg:
            free_maskes = get_free_mask(
                pos_bl              = tensorf.pos_bit_length, 
                view_bl             = tensorf.view_bit_length, 
                fea_bl              = tensorf.fea_bit_length, 
                den_bl              = tensorf.density_n_comp,
                app_bl              = tensorf.app_n_comp,
                using_decomp_mask   = args.free_decomp,
                total_step          = args.n_iters, 
                step                = iteration, 
                ratio               = ratio,
                device              = device
            )
        else:
            free_maskes = get_free_mask()


        # ==> Render from grid
        # ================================
        (rgb_map, alphas_map, depth_map, weights, uncertainty, num_samples) = renderer(
            rays_train,
            tensorf,
            free_maskes,
            chunk=args.batch_size,
            N_samples=nSamples,
            white_bg=white_bg,
            ndc_ray=ndc_ray,
            device=device,
            is_train=True,
        )


        # ==> Loss
        # ================================
        total_loss = loss = torch.mean((rgb_map - rgb_train) ** 2)

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


        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()


        loss = loss.detach().item()
        PSNRs.append(-10.0 * np.log(loss) / np.log(10.0))
        summary_writer.add_scalar(
            "train/PSNR", PSNRs[-1], global_step=iteration
        )
        summary_writer.add_scalar(
            "train/mse", loss, global_step=iteration
        )


        # ==> Update learning rate
        # ================================
        for param_group in optimizer.param_groups:
            param_group["lr"] = param_group["lr"] * lr_factor


        # ==> Print loss
        # ================================
        if iteration % args.progress_refresh_rate == 0:
            step_toc = time.time()
            pbar.set_description(
                f"Iteration {iteration:05d}:"
                + f" train_psnr = {float(np.mean(PSNRs)):.2f}"
                + f" test_psnr = {float(np.mean(PSNRs_test)):.2f}"
                + f" mse = {loss:.6f}"
                + f" elapsed_time = {step_toc - run_tic:.2f}"
            )
            PSNRs = []


        # ==> PSNRs metric
        # ================================
        if iteration % args.train_vis_every == 0:
            if iteration % args.vis_every == 0:
                PSNRs_test = PSNRs_calculate(
                    tensorf,
                    test_dataset,
                    renderer, 
                    chunk=args.batch_size,
                    N_samples=nSamples,
                    white_bg=white_bg, 
                    ndc_ray=ndc_ray,                   
                    device=device
                )
            history['iteration'].append(iteration)
            history['train_psnr'].append(round(float(np.mean(PSNRs_train)), 2))
            history['test_psnr'].append(round(float(np.mean(PSNRs_test)), 2))
            history['mse'].append(round(loss, 5))


            # ==> Save rendered image
            # ================================
            save_rendered_image_per_train(
                train_dataset       = train_gift_data,
                test_dataset        = test_gift_data,
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
            PSNRs_train = []

        return 
        # ==> Update alphaMask list
        # ================================
        if iteration in update_AlphaMask_list:
            if (reso_cur[0] * reso_cur[1] * reso_cur[2]) < 256**3:
                reso_mask = reso_cur

            new_aabb = tensorf.updateAlphaMask(tuple(reso_mask), free_maskes['decomp']['den'])
            if iteration == update_AlphaMask_list[0]:
                tensorf.shrink(new_aabb)

            # Filter rays outside the bbox
            if not ndc_ray and iteration == update_AlphaMask_list[1]:
                allrays, allrgbs = tensorf.filtering_rays(allrays, allrgbs)
                trainingSampler = SimpleSampler(
                    allrgbs.shape[0], batch_size
                )



        if iteration in upsamp_list:
            if len(upsamp_list) == len(mask_ratio_list):
                ratio = mask_ratio_list[upsamp_list.index(iteration)]

            n_voxels = N_voxel_list.pop(0)
            reso_cur = N_to_reso(n_voxels, tensorf.aabb)
            nSamples = min(nSamples, cal_n_samples(reso_cur, step_ratio))            
            tensorf.upsample_volume_grid(reso_cur)

            if lr_upsample_reset: lr_scale = 1  # 0.1 ** (iteration / args.n_iters)
            else:lr_scale = lr_decay_target_ratio ** (iteration / n_iters)

            grad_vars = tensorf.get_optparam_groups(lr_init * lr_scale, lr_basis * lr_scale)
            optimizer = torch.optim.Adam(grad_vars, betas=(0.9, 0.99))

        if iteration in save_ckpt_every:
            tensorf.save(f'{logfolder}/{iteration//1000}k_{expname}.th')        

        return

    tensorf.save(f'{logfolder}/final_{args.expname}.th')
    elapsed_time = time.time() - run_tic
    np.savetxt(f"{logfolder}/training_time.txt", np.asarray([elapsed_time]))
    print(f"Total time {elapsed_time:.2f}s.")

    if args.render_train:
        os.makedirs(f"{logfolder}/imgs_train_all", exist_ok=True)
        train_dataset = dataset(args.datadir, split="train", downsample=args.downsample_train, is_stack=True)

        PSNRs_test = evaluation(
            train_dataset,
            tensorf,
            renderer,
            f"{logfolder}/imgs_train_all/",
            N_vis=-1,
            N_samples=-1,
            white_bg=white_bg,
            ndc_ray=ndc_ray,
            device=device,
        )
        print(f"======> {args.expname} test all psnr: {np.mean(PSNRs_test)} <========================")

    if render_test:
        os.makedirs(f"{logfolder}/imgs_test_all", exist_ok=True)
        PSNRs_test = evaluation(
            test_dataset,
            tensorf,
            renderer,
            f"{logfolder}/imgs_test_all/",
            N_vis=-1,
            N_samples=-1,
            white_bg=white_bg,
            ndc_ray=ndc_ray,
            device=device,
        )
        summary_writer.add_scalar("test/psnr_all", np.mean(PSNRs_test), global_step=iteration)
        print(f"======> {args.expname} test all psnr: {np.mean(PSNRs_test)} <========================")

    if args.render_path:
        c2ws = test_dataset.render_path        
        print("========>", c2ws.shape)
        os.makedirs(f"{logfolder}/imgs_path_all", exist_ok=True)
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

    ckpt_path = reconstruction()

    """args = config_parser()

    if args.export_mesh:
        export_mesh(args, args.ckpt)

    if args.render_only and (args.render_test or args.render_path or args.render_train):
        render_test(args)
    elif args.config:
        ckpt_path = reconstruction(args)        
        export_mesh(args, ckpt_path)  

        import shutil 
        shutil.copy"""