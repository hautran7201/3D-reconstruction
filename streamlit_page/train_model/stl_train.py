import streamlit as st
import numpy as np
import os
import glob
from streamlit_page.train_model.stl_data_preparation import *
from streamlit_page.train_model.stl_data_visualization import *
from streamlit_page.train_model.stl_obj_visualization import *
from streamlit_page.train_model.stl_training import *
def train_model():
    st.header('Data preparation')
    
    prepare_data, training, vis_image, vis_obj_3d = st.tabs(["Create config", "Training", "Visualize image", 'Visualize 3d object'])
    
    data_folder = os.path.join('data','data')
    os.makedirs(data_folder, exist_ok=True)    
    config_folder = os.path.join('data','config')
    os.makedirs(config_folder, exist_ok=True)
    obj_folder = os.path.join('data','object_data')
    os.makedirs(obj_folder, exist_ok=True)
    log_folder = os.path.join('few_nerf', 'log')
    os.makedirs(obj_folder, exist_ok=True)

    with prepare_data:
        data_preparation(data_folder, config_folder)
    
    with training:
        training_model(config_folder)

    with vis_image:
        data_visualization(data_folder)
    
    with vis_obj_3d:
        obj_visualization(obj_folder)