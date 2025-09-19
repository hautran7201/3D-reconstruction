import streamlit as st
import os
import json
import pandas as pd
import numpy as np 
from collections import defaultdict
from few_nerf.opt import *
from few_nerf.train import reconstruction


def training_model(config_dir):
    

    config_option = st.selectbox(
        "Select config", 
        os.listdir(config_dir), 
        key='config_option', 
        index=None, 
    )

    if config_option != None:
        config_path = os.path.join(config_dir, config_option)
        cmd_arguments = [
            '--config',
            config_path
        ]

        args = config_parser(cmd_arguments)
        with open(os.path.join(args.datadir, f"transforms_train.json"), 'r') as f:
            meta = json.load(f)
        
        if args.train_idxs:
            train_idxs = args.train_idxs
        else:
            train_idxs = range(0, len(meta['frames'])-1)

        image_list = defaultdict(dict)
        for i in train_idxs:#img_list:#

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

            if args.dataset_name == "blender":
                file_path = frame['file_path'].split('.')[-1]
                image_path = os.path.join(args.datadir, f"{frame['file_path']}.png")
            if args.dataset_name == "human":
                file_path = frame['file_path'].split('\\')[-1].split('.')[-2]
                image_path = os.path.join(args.datadir, "train",file_path+'.png')
            formatted_path = os.path.join(*image_path.split('/'))
            
            file_path = formatted_path

            file_name = formatted_path.split('\\')[-1]
            image_list[file_name] = {
                'camera_angles': camera_angles,
                'transform_frames': transform_frames, 
                'file_path': file_path
            }

        st.markdown('#')
        st.markdown('#')
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

        st.divider()
        train_button = st.button(
            'Start training',
            key = 'train_button'
        )        

        if train_button:    
            with st.spinner('Wait for it...'):
                ckpt_path = reconstruction(args)

                import shutil 
                shutil.copy(args.config, ckpt_path[:-3]+'.txt')    