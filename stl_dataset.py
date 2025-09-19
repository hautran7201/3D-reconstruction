import pyvista as pv
import streamlit as st
import os 
from stpyvista import stpyvista

def dataset_mode():
    province_box, feature_box = st.columns(2)

    # Selected province box
    with province_box:
        province = st.selectbox(
            "Province",
            list(st.session_state.provinces),
            key='province_dataset_box'
        )
    """## Initialize a plotter object
    pv.global_theme.show_scalar_bar = False
    plotter = pv.Plotter(window_size=[400,400])

    ## Create a mesh with a cube 
    
    file_name = r'd:\Desktop\Downloads\lego (23).ply'
    # mesh = pv.Cube(center=(0,0,0))
    mesh = pv.read(file_name)
    
    st.title(os.path.splitext(os.path.basename(file_name))[0])    

    ## Add some scalar field associated to the mesh
    mesh['myscalar'] = mesh.points[:, 2]*mesh.points[:, 0]

    ## Add mesh to the plotter
    plotter.add_mesh(mesh, scalars='myscalar', cmap='binary', line_width=1)

    ## Final touches
    plotter.view_isometric()
    plotter.background_color = 'white'

    ## Send to streamlit
    stpyvista(plotter, key="pv_cube")"""