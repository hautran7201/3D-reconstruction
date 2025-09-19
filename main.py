import streamlit as st
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import pyvista as pv
from stpyvista import stpyvista
sys.path.insert(5, './')
from stl_dataset import dataset_mode
from streamlit_page.train_model.stl_train import *
from streamlit_page.test_model.stl_test import *
from streamlit_page.inference_model.stl_inference import *
# Side bar
with st.sidebar:
    st.markdown('# **FEW-TENSORF**')
    option = st.selectbox(
        'Select option',
        ['Train model', 'Test model','Inference'],
        key='select_option'
    )

log_folder = os.path.join('few_nerf', 'log')

# Option
if option == 'Train model':
    train_model()
elif option == 'Test model':
    test_model()
elif option == 'Inference':
    inference(log_folder)