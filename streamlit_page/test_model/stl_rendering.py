import streamlit as st
import os
import glob
from few_nerf.opt import *
from few_nerf.train import render_test

NUM_OUTPUT = 10

def rendering(log_dir):
    object_option = st.selectbox(
        "Select object", 
        os.listdir(log_dir), 
        key='object_option', 
        index=None,
    )

    data_path = 'data/data'
    dataset_type = st.selectbox(
        "Select data Type", 
        os.listdir(data_path), 
        key='dataset_type', 
        index=None,
    )

    datadir = ''
    if dataset_type:
        dataset_folder = st.selectbox(
            "Select data folder", 
            os.listdir(os.path.join(data_path, dataset_type)), 
            key='dataset_folder', 
            index=None,
        )   
        if dataset_folder:
            datadir = os.path.join(data_path, dataset_type, dataset_folder)

            render_test_box = st.selectbox(
                "Choice render test", 
                [True, False], 
                key='render_test_box', 
            )

            render_train_box = st.selectbox(
                "Choice render train", 
                [True, False], 
                key='render_train_box'
            )

            if object_option:
                cmd_arguments = [
                    '--config',
                    os.path.join(log_dir, object_option, 'final_'+object_option+'.txt'),
                    '--ckpt',
                    os.path.join(log_dir, object_option, 'final_'+object_option+'.th'),
                    '--datadir',
                    datadir,
                    '--render_only',
                    '1',
                    '--render_test',
                    '1' if render_test_box else '0',
                    '--render_train',
                    '1' if render_train_box else '0',
                    '--N_train_imgs',
                    str(NUM_OUTPUT), 
                    '--N_test_imgs',
                    str(NUM_OUTPUT),                     
                ]

                args = config_parser(cmd_arguments)
                
                render_button = st.button(
                    "Rendering",
                    key = 'render_button'
                )

                if render_button:
                    with st.spinner('Wait for it...'):
                        render_test(args)