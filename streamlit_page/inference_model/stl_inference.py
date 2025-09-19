import streamlit as st
import os
# import glob
import json
import pandas as pd
import numpy as np 
from collections import defaultdict
from few_nerf.opt import *
from few_nerf.renderer import *
from few_nerf.dataLoader import dataset_dict
from streamlit_image_comparison import image_comparison
from PIL import Image
# from few_tensorf.train import render_test
import torch

from few_nerf.train import export_mesh
import pyvista as pv
from stpyvista import stpyvista
import re 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

renderer = OctreeRender_trilinear_fast

# st.set_page_config(page_title="Image-Comparison Example", layout="centered")

@torch.no_grad()
def evaluate(test_dataset,tensorf, args, renderer, savePath=None, N_vis=5, chunk=4096, N_samples=-1,
               white_bg=False, ndc_ray=False, compute_extra_metrics=True, device='cuda'):
    PSNRs, rgb_maps, depth_maps = [], [], []
    ssims,l_alex,l_vgg=[],[],[]
    os.makedirs(savePath, exist_ok=True)
    gt_path = os.path.join(savePath, 'gt') 
    os.makedirs(gt_path, exist_ok=True)
    pred_path = os.path.join(savePath, 'pred') 
    os.makedirs(pred_path, exist_ok=True)
    rgbd_path = os.path.join(savePath, 'rgbd') 
    os.makedirs(rgbd_path, exist_ok=True)

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

        rgb_map, _, depth_map, _, _, _ = renderer(rays, tensorf, chunk=chunk, N_samples=N_samples,
                                        ndc_ray=ndc_ray, white_bg = white_bg, device=device)
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
        # rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
        gt_rgb = (gt_rgb.numpy() * 255).astype('uint8')
        rgb_maps.append(rgb_map)
        depth_maps.append(depth_map)
        if savePath is not None:
            imageio.imwrite(f'{savePath}/gt/{idx:03d}.png', gt_rgb)
            imageio.imwrite(f'{savePath}/pred/{idx:03d}.png', rgb_map)
            rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
            imageio.imwrite(f'{savePath}/rgbd/{idx:03d}.png', rgb_map)

    if PSNRs:
        psnr = np.mean(np.asarray(PSNRs))
        if compute_extra_metrics:
            ssim = np.mean(np.asarray(ssims))
            l_a = np.mean(np.asarray(l_alex))
            l_v = np.mean(np.asarray(l_vgg))
            np.savetxt(f'{savePath}/mean.txt', np.asarray([psnr, ssim, l_a, l_v]))
        else:
            np.savetxt(f'{savePath}/mean.txt', np.asarray([psnr]))


    return PSNRs

@torch.no_grad()
def rendering(args, split = 'test', index =1):
    # init dataset
    dataset = dataset_dict[args.dataset_name]

    inference_data = dataset(args.datadir, split=split, downsample=args.downsample_train, is_stack=True, tqdm=True, indexs=[index])

    white_bg = inference_data.white_bg
    ndc_ray = args.ndc_ray

    if not os.path.exists(args.ckpt):
        print('the ckpt path does not exists!!')
        return

    ckpt = torch.load(args.ckpt, map_location=device)
    kwargs = ckpt['kwargs']
    kwargs.update({'device': device})
    tensorf = eval(args.model_name)(args, **kwargs)
    tensorf.load(ckpt)

    logfolder = os.path.dirname(args.ckpt)
    inference_folder = f'{logfolder}/imgs_inference/'

    os.makedirs(inference_folder, exist_ok=True)
    psnr = evaluate(inference_data,tensorf, args, renderer, inference_folder,
                            N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)
    psnr = np.mean(psnr)
    print(f'======> {args.expname} PSNR: {psnr} <========================')
    return psnr , inference_folder


def inference(log_dir):
    st.title("Inference")
    st.markdown('#')

    """log_dir = os.path.join('few_tensorf','log')
    os.makedirs(log_dir, exist_ok=True)"""

    model_option = st.selectbox(
        "Select model:", 
        os.listdir(log_dir), 
        key='model_option', 
        index=None,
    )

    data_root = os.path.join('data','data')
    dataset_type = st.selectbox(
        "Select dataset type", 
        os.listdir(data_root), 
        key='dataset_type', 
        index=None,
    )


    if dataset_type:
        dataset_path = os.path.join(data_root,dataset_type)
        scene = st.selectbox(
            "Select scene", 
            os.listdir(dataset_path), 
            key='scene', 
            index=None,
        )


        # # data_root = 'data/data'
        # if scene:
        split_type = st.selectbox(
            "Select Split:", 
            ['train','test'], 
            key='split_type', 
            index=None,
        )

    if model_option and dataset_type and scene and split_type:
        config_path = os.path.join(log_dir, model_option,f'final_{model_option}.txt')
        cmd_arguments = [
            '--config',
            config_path,
            '--ckpt',
            os.path.join(log_dir, model_option, 'final_'+model_option+'.th'),
            # '--datadir',
            # datadir,
            '--render_only',
            '1',
            f'--render_{split_type}',
            '1'
        ]

        args = config_parser(cmd_arguments)


        # split_pattern = re.compile(r"\\")
        # if  split_pattern.search(args.datadir):
        #     args.datadir = "/".join(args.datadir.split('\\'))

        args.datadir = os.path.join(dataset_path, scene)

        js_path = os.path.join(args.datadir , f"transforms_{split_type}.json")

        js_path = os.path.join(*js_path.split('/'))
        with open(js_path, 'r') as f:
            meta = json.load(f)

        # render_idxs = range(1,50)
        render_idxs = range(0, len(meta['frames'])-1)

        image_list = defaultdict(dict)
        for i in render_idxs:
            frame = meta['frames'][i]
            transform_frames = frame['transform_matrix']

            camera_angle_x = 0
            if 'camera_angle_x' in meta:
                camera_angle_x = meta['camera_angle_x']
            camera_angle_y = 0
            if 'camera_angle_y' in meta:
                camera_angle_y = meta['camera_angle_y']
            if camera_angle_x == 0:
                camera_angle_x = camera_angle_y
            if camera_angle_y == 0:
                camera_angle_y = camera_angle_x
            camera_angles = [camera_angle_x, camera_angle_y]

            if args.dataset_name == "blender" and dataset_type == "nerf_synthetic":
                file_path = frame['file_path'].split('.')[-1]
                image_path = os.path.join(args.datadir, f"{frame['file_path']}.png")
            elif args.dataset_name == "human" and dataset_type == "Human":
                file_path = frame['file_path'].split('\\')[-1].split('.')[-2]
                image_path = os.path.join(args.datadir , split_type,file_path+'.png')
            else:
                if args.dataset_name == "human":
                    raise ValueError("you must choose dataset type 'Human'")
                if args.dataset_name == "blender":
                    raise ValueError("you must choose dataset type 'nerf_synthetic'")
            formatted_path = os.path.join(*image_path.split('/'))
            
            file_path = formatted_path

            file_name = formatted_path.split('\\')[-1]
            image_list[file_name] = {
                'camera_angles': camera_angles,
                'transform_frames': transform_frames, 
                'file_path': file_path,
                'index': i
            }

        st.markdown('#')
        st.header('Choose your input')
        st.divider()
        col1, col2 = st.columns([0.3, 0.7])
        col1.subheader('List of images:')
        

        train_image_choice = col2.selectbox(
            "List of images", 
            image_list.keys(), 
            key='train_image_choice',              
            label_visibility="collapsed"
        )

        if train_image_choice:
            image_dict = image_list[train_image_choice]
            col1,col2 = st.columns([0.43, 0.57])

            camera_angle = image_dict['camera_angles']
            transform_matrix = np.round(np.array(image_dict['transform_frames']), decimals=4)
            transform_matrix_pd = pd.DataFrame(transform_matrix)
            col1.markdown('#')
            col1.write('CAMERA ANGLE')
            col1.write(camera_angle)
            col1.write('TRANSFORM MATRIX')
            col1.write(transform_matrix_pd)

            # Select photo a send it to button
            photo = image_dict['file_path']
            col2.image(photo,caption=photo)

        render_button = st.button(
            "Rendering",
            key = 'render_button'
        )

        if render_button:
            with st.spinner('Wait for it...'):
                psnr, inference_folder = rendering(args, split=split_type, index= image_list[train_image_choice]["index"])
            # inference_folder = 'few_tensorf/log/ficus/imgs_inference'

            st.divider()
            st.header('Inference Result')
            st.subheader('PSNR score: ' + str(psnr.round(2)))
            compare, mesh = st.tabs(["Visual Compare", "3D Mesh"])

            with compare:
                    pred_image_folder = os.path.join(inference_folder,"pred")
                    gt_image_folder = os.path.join(inference_folder,"gt")

                # try:
                    pred_image_paths = os.listdir(pred_image_folder)
                    # image_metric, image_tab = st.tabs(['Metric', 'Image'])

                    # with image_metric:
                    #     history = dict(numpy.load(os.path.join(log_dir, model_option, 'history.npz')))
                    #     # history_pd = pd.DataFrame.from_dict(history)
                    #     st.line_chart(history, x='iteration', y=['train_psnr', 'test_psnr'])

                    # with image_tab:
                    image_choice = "000.png"
                    if image_choice:
                        pred_image = Image.open(os.path.join(pred_image_folder, image_choice))
                        gt_image = Image.open(os.path.join(gt_image_folder, image_choice))

                        image_comparison(
                            img1=gt_image,
                            img2=pred_image,
                            label1="Ground truth",
                            label2="Prediction",
                            show_labels=True,
                            make_responsive=True,
                            in_memory=True,
                        )        
                # except:
                #     st.error("The predicted image folder doesn't exist", icon="🚨")
            
            with mesh:
                ckpt_path = os.path.join(log_dir, model_option,f'final_{model_option}.th')
                obj_file = ckpt_path[:-3]+'.ply'
                if os.path.exists(obj_file):
                    pass
                else:        
                    export_mesh(args, ckpt_path)

                pv.global_theme.show_scalar_bar = False
                plotter = pv.Plotter(notebook= True, window_size=[400,400])
                
                mesh = pv.read(obj_file)
                
                # st.title(os.path.splitext(os.path.basename(file_name))[0])    

                ## Add some scalar field associated to the mesh
                mesh['myscalar'] = mesh.points[:, 2]*mesh.points[:, 0]
                # tex = pv.read_texture(tex_file)

                ## Add mesh to the plotter
                plotter.add_mesh(mesh, scalars='myscalar', cmap='binary', line_width=1)

                ## Final touches
                plotter.view_isometric()
                plotter.background_color = 'white'

                ## Send to streamlit
                stpyvista(plotter, key="pv_cube")