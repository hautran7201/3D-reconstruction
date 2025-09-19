import streamlit as st
from few_nerf.opt import *
import os
import glob
from few_nerf.train import export_mesh
import pyvista as pv
from stpyvista import stpyvista


def mesh_extract(log_dir):
    object_mesh_option = st.selectbox(
        "Select object", 
        os.listdir(log_dir), 
        key='object_mesh_option', 
        index=None,
    )

    if object_mesh_option:
        ckpt_path = os.path.join(log_dir, object_mesh_option, 'final_'+object_mesh_option+'.th')

        cmd_arguments = [
            '--export_mesh',
            '1'
        ]
        
        args = config_parser(cmd_arguments)

        extract_button = st.button(
            'Extract mesh',
            key = 'extract_button'
        )

        if extract_button:
            # ckpt_path = os.path.join(log_dir, model_option,f'final_{model_option}.th')
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



