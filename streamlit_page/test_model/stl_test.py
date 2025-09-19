import streamlit as st
import numpy as np
import os
import glob
import configargparse
from few_nerf.dataLoader.__init__ import *
from few_nerf.opt import *
from streamlit_page.test_model.stl_rendering import *
from streamlit_page.test_model.stl_mesh_extract import *
from streamlit_page.test_model.stl_comparation import *

def test_model():
    st.header('Testing model')

    rendering_tab, mesh_extract_tab, comparation_tab = st.tabs(["Rendering", "Mesh extract", "Comparison"])

    log_dir = os.path.join('few_nerf','log')
    with rendering_tab:
        rendering(log_dir)
    
    with mesh_extract_tab:
        mesh_extract(log_dir)

    with comparation_tab:
        comparation(log_dir)

            

