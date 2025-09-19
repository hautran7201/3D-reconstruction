import pyvista as pv
import streamlit as st
import os 
from stpyvista import stpyvista


def obj_visualization(obj_dir):

    obj_dataset_list = os.listdir(obj_dir)

    object_option = st.selectbox(
        "Select object", 
        obj_dataset_list + ['log'], 
        key='object_option', 
        index=None,
    )

    if object_option:
        ## Initialize a plotter object
        pv.global_theme.show_scalar_bar = False
        plotter = pv.Plotter(window_size=[400,400])

        if object_option == 'log':
            obj_dir = 'few_nerf'
            file_tail = '.ply'
            file_head = 'final_'
        else:
            file_head = ''
            file_tail = '.obj'

        object_name = st.selectbox(
            "Select object name", 
            os.listdir(os.path.join(obj_dir, object_option)), 
            key='object_name', 
            index=None,
        )
        
        if object_name:
            obj_file = os.path.join(obj_dir, object_option, object_name, file_head+object_name+file_tail)                
            # mesh = pv.Cube(center=(0,0,0))

            mesh = pv.read(obj_file)
            
            # st.title(os.path.splitext(os.path.basename(file_name))[0])    

            ## Add some scalar field associated to the mesh
            mesh['myscalar'] = mesh.points[:, 2]*mesh.points[:, 0]
            # tex = pv.read_texture(tex_file)

            ## Add mesh to the plotter
            plotter.add_mesh(mesh, scalars='myscalar', cmap='gray', line_width=1)

            ## Final touches
            plotter.view_isometric()
            plotter.background_color = 'white'

            ## Send to streamlit
            stpyvista(plotter, key="pv_cube")