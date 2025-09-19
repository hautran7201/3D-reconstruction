import streamlit as st
import os
import glob
import time
from  utils import * 
from few_nerf.dataLoader.__init__ import *


def data_preparation(data_dir, config_dir):
    
    st.subheader('Folder setting')

    choice_dataset = st.selectbox(
        "Select folder dataset", 
        os.listdir(data_dir), 
        key='choice_dataset', 
        index=None,
        placeholder="Dataset"
    )

    if choice_dataset:

        dataset_folder = os.path.join(data_dir, choice_dataset)
        obj_list = get_folder_names(dataset_folder)

        training_data_dir = ''
        if len(obj_list) == 0:
            pass
        else:
            dataset_obj_option = st.selectbox(
                "Select dataset object", 
                obj_list, 
                key='train_dataset_obj_option', 
                index=None,
                placeholder="Object"
            )
            
            if dataset_obj_option:
                training_data_dir = os.path.join(data_dir, choice_dataset, dataset_obj_option)

    exp_name_box = st.text_input(
        'Export name',       
        key='exp_name_box',
    )

    over_write_box = st.selectbox(
        "Overwrite", 
        [True, False], 
        key='over_write_box'        
    )


    st.markdown('#')
    st.divider()
    st.subheader('Dataloader')

    dataloader_box = st.selectbox(
        "Select folder dataset", 
        dataset_dict.keys(), 
        key='train_dataloader_box', 
        index=None,
        placeholder="Data loader"
    )



    st.markdown('#')
    st.divider()
    st.subheader('Data Selection')
    
    train_selection_toggle = st.toggle('Random index train', value=True)
    rd_train_selection_box = 0
    tx_train_selection_box = []
    if train_selection_toggle:
        rd_train_selection_box = st.number_input(
            'Train data', 
            min_value=0, 
            step=1, 
            key='train_selection_box', 
            placeholder='Number of random images', 
            label_visibility="collapsed"
        )
    else:
        tx_train_selection_box = st.text_input(
            'Train data',       
            key='train_slection_box', 
            placeholder='List of training indexs', 
            label_visibility="collapsed"
        )
        
    val_selection_toggle = st.toggle('Random index validation', value=True) 
    rd_val_selection_box = 0
    tx_val_selection_box = []
    if val_selection_toggle:
        rd_val_selection_box = st.number_input(
            'Validation data', 
            min_value=0, 
            step=1, 
            key='val_selection_box', 
            placeholder='Number of random images', 
            label_visibility="collapsed"
        )
    else:
        tx_val_selection_box = st.text_input(
            'Validation data',       
            key='val_slection_box', 
            placeholder='List of val indexs', 
            label_visibility="collapsed"
        )

    test_selection_toggle = st.toggle('Random index test', value=True) 
    rd_test_selection_box = 0
    tx_test_selection_box = []
    if test_selection_toggle:
        rd_test_selection_box = st.number_input(
            'Test data', 
            min_value=0, 
            step=1, 
            key='test_selection_box', 
            placeholder='Number of random images', 
            label_visibility="collapsed"
        )
    else:
        tx_test_selection_box = st.text_input(
            'Test data',       
            key='test_slection_box', 
            placeholder='List of test indexs', 
            label_visibility="collapsed"
        )


    st.markdown('#')
    st.divider()
    st.subheader('Training time')

    iteration_col, batch_size_col, samples_col = st.columns(3)
    with iteration_col:
        iteration_box = st.number_input(
            'Number of iterations', 
            value=15000,
            min_value=0, 
            step=1, 
            key='iteration_box', 
        )
    with batch_size_col:
        batch_size_box = st.number_input(
            'Batch size', 
            min_value=0, 
            value=1024,
            step=1, 
            key='batch_size_box', 
        )
    with samples_col:
        ratio_step_box = st.number_input(
            'Ratio sample step', 
            value=0.5,
            min_value=0.0, 
            step=1.0, 
            key='ratio_step_box', 
        )

    st.markdown('#')
    st.divider()
    st.subheader('Resolution')

    N_voxel_init_col, N_voxel_final_col = st.columns(2)
    with N_voxel_init_col:
        N_voxel_init_box = st.number_input(
            'Number voxel init', 
            min_value=0, 
            value=128,
            step=1, 
            key='N_voxel_init_box', 
        )
        downsample_box = st.number_input(
            'Downsample image', 
            min_value=0, 
            value=2,
            step=1, 
            key='downsample_box', 
        )

    with N_voxel_final_col:
        N_voxel_final_box = st.number_input(
            'Number voxel final', 
            min_value=0, 
            value=300,
            step=1, 
            key='N_voxel_final_box', 
        )

    
    st.markdown('#')
    st.divider()
    st.subheader('Model name')

    decomposition_model_col, mlp_model_col = st.columns(2)
    with decomposition_model_col:
        decomposition_model_box = st.selectbox(
            "Select decomposition model", 
            ['TensorVMSplit', 'TensorCP'], 
            key='decomposition_model_box', 
            index=None,
            placeholder="Model"
        )
    with mlp_model_col:
        mlp_model_box = st.selectbox(
            "Select mlp model", 
            ['MLP_Fea', 'MLP_PE', 'MLP'], 
            key='mlp_model_box', 
            index=None,
            placeholder="Model"
        )


    st.markdown('#')
    st.divider()
    st.subheader('Utils')

    test_col1, test_col2, test_col3 = st.columns(3)
    with test_col1:
        train_vis_every_box = st.number_input(
            'Number of iterations to train visualization', 
            min_value=0, 
            value=1000,
            step=1, 
            key='train_vis_every_box', 
        )
    with test_col2:
        test_vis_every_box = st.number_input(
            'Number of iterations to test visualization', 
            min_value=0, 
            value=1000,
            step=1, 
            key='test_vis_every_box', 
        )
    with test_col3:
        save_ckpt_every_box = st.text_input(
            'Number of iterations to save checkpoint',       
            key='save_ckpt_every_box',
            value=[15000, 30000],
            placeholder='List'
        )
    

    st.markdown('#')
    st.divider()
    st.subheader('Decomposition')

    density_col, appearance_col = st.columns(2)
    with density_col:
        density_box = st.number_input(
            'Number of density components', 
            min_value=0, 
            value=16,
            step=1, 
            key='density_box', 
        )
    with appearance_col:
        appearance_box = st.number_input(
            'Number of appearance components', 
            min_value=0, 
            value=48,
            step=1, 
            key='appearance_box', 
        )


    st.markdown('#')
    st.divider()
    st.subheader('Feature')

    feature_col1, feature_col2, feature_col3, feature_col4 = st.columns(4)
    with feature_col1:
        position_enc_box = st.number_input(
            'Position bandwidth', 
            min_value=0, 
            value=2,
            step=1, 
            key='position_enc_box', 
        )
    with feature_col2:
        view_enc_box = st.number_input(
            'View direction bandwidth',  
            min_value=0, 
            value=2,
            step=1, 
            key='view_enc_box', 
        )
    with feature_col3:
        fea_enc_box = st.number_input(
            'Feature bandwidth',  
            min_value=0, 
            value=2,
            step=1, 
            key='fea_enc_box', 
        )
    with feature_col4:
        fea_dim_box = st.number_input(
            'Feature dimension',   
            min_value=0, 
            value=27,
            step=1, 
            key='fea_dim_box', 
        )


    st.markdown('#')
    st.divider()
    st.subheader('Free nerf')
 
    free_tab, occlusion_tab = st.tabs(["Free mask", "Occlusion loss"])
    with free_tab:
        free_col1, free_col2 = st.columns([0.7, 0.3])
        with free_col1:
            free_mask_box = st.selectbox(
                'Free mask',
                [True, False],
                key='free_mask_box'
            )
            freq_reg_ratio_box = st.number_input(
                'Frequence ratio',   
                min_value=0.0, 
                value=0.8,
                max_value=1.0,                 
                key='freq_reg_ratio_box', 
            )
    with occlusion_tab:
        occ_col1, occ_col2 = st.columns([0.7, 0.3])
        with occ_col1:
            occ_weight_box = st.number_input(
                'Occlusion weight loss',
                min_value=0, 
                max_value=1, 
                key='occ_weight_box'
            )
            occ_reg_range_box = st.number_input(
                'Occlusion regularization range',
                min_value=0, 
                step=1,
                key='occ_reg_range_box'
            )
            occ_wb_range_box = st.number_input(
                'Occlusion white black range',
                min_value=0, 
                step=1,
                key='occ_wb_range_box'
            )
            occ_wb_prior_box = st.selectbox(
                'Occlusion white black prior',
                [True, False],
                index=None,
                key='occ_wb_prior_box'
            )

    
    st.markdown('#')
    st.divider()
    st.subheader('Regularization')

    l1_tab, ortho_tab, tv_tab, alpha_tab = st.tabs(['L1', 'Ortho', 'Tv', 'Alpha'])
    with l1_tab:
        l1_col1, _ = st.columns([0.7, 0.3])
        with l1_col1:
            L1_weight_inital_box = st.number_input(
                'L1 weight inital',
                value=8e-5,
                min_value=0.0, 
                key='L1_weight_inital_box'
            )
            L1_weight_rest_box = st.number_input(
                'L1 weight rest',
                value=4e-5,
                min_value=0.0, 
                key='L1_weight_rest_box'
            )
    with ortho_tab:
        ortho_col1, _ = st.columns([0.7, 0.3])
        with ortho_col1:
            ortho_weight_box = st.number_input(
                'Ortho weight',
                value=0.01,
                min_value=0.0, 
                max_value=1.0, 
                key='ortho_weight_box'
            )
    with tv_tab:
        tv_col1, _ = st.columns([0.7, 0.3])
        with tv_col1:
            tv_weight_density_box = st.number_input(
                'TV weight density',
                value=0.01,
                min_value=0.0, 
                max_value=1.0,    
                key='tv_weight_density_box'
            )
            TV_weight_app_box = st.number_input(
                'TV weight app',
                value=0.01,
                min_value=0.0, 
                max_value=1.0,    
                key='TV_weight_app_box'
            )
    with alpha_tab:
        alpha_col1, _ = st.columns([0.7, 0.3])
        with alpha_col1:
            rm_weight_mask_thre_box = st.number_input(
                'Remove weight mask threshold',
                value=1e-4,
                min_value=0.0,  
                max_value=1.0,     
                key='rm_weight_mask_thre_box'
            )
            alpha_mask_thre_box = st.number_input(
                'Alpha mask threshold',
                value=1e-4,
                min_value=0.0, 
                max_value=1.0,    
                key='alpha_mask_thre_box'
            )


    st.markdown('#')
    st.divider()
    st.subheader('Save config')

    config_name_box = st.text_input(
        'Config name',       
        value=None, 
        key='config_name_box',
    )

    save_cf_button = st.button('Save config', type="primary")

    if config_name_box and save_cf_button:
        with open(os.path.join(config_dir, config_name_box) , "w") as file:
            
            file.write(
f'''
#------ Folder ------ 
dataset_name = {dataloader_box}
datadir = {training_data_dir}
expname = {exp_name_box}
basedir = {os.path.join("few_nerf","log")}

#------ Number images ------
train_idxs    = {tx_train_selection_box}
val_idxs      = {tx_val_selection_box}
test_idxs     = {tx_test_selection_box}

N_train_imgs  = {rd_train_selection_box}
N_test_imgs   = {rd_test_selection_box}


#------ Config parameters ------
n_iters = {iteration_box}
batch_size = {batch_size_box}
step_ratio = {ratio_step_box}


#------ Resolution ------
occ_grid_reso           = 0
N_voxel_init            = {N_voxel_init_box**3} # 128**3 2097156
N_voxel_final           = {N_voxel_final_box**3} # 300**3
upsamp_list             = [2000,3000,4000,5500,7000]
update_AlphaMask_list   = [2000,4000]
downsample_train        = {downsample_box}


#------ Model name ------
model_name    = {decomposition_model_box}
shadingMode   = {mlp_model_box}
fea2denseAct  = softplus
overwrt       = {over_write_box} 


#------ Test ------
N_vis             = 5
vis_every         = {test_vis_every_box}
train_vis_every   = {train_vis_every_box}
save_ckpt_every   = {save_ckpt_every_box}


#------ Decomposition ------
n_lamb_sigma = {[density_box, density_box, density_box] if 'VM' in decomposition_model_box else density_box}
n_lamb_sh = {[appearance_box, appearance_box, appearance_box] if 'VM' in decomposition_model_box else appearance_box}


#------ Feature config ------
pos_pe  = {position_enc_box}
view_pe = {view_enc_box}
fea_pe = {fea_enc_box}
data_dim_color = {fea_dim_box}


#------ Frequence reg ------
free_reg        = {free_mask_box}
freq_reg_ratio  = {freq_reg_ratio_box}


#------ Occlusion reg ------
occ_reg_loss_mult   = {occ_weight_box}
occ_reg_range       = {occ_reg_range_box}
occ_wb_range        = {occ_wb_range_box}
occ_wb_prior        = {occ_wb_prior_box}


#------ Reg ------
L1_weight_inital    = {L1_weight_inital_box}
L1_weight_rest      = {L1_weight_rest_box}
Ortho_weight        = {ortho_weight_box}
TV_weight_density   = {tv_weight_density_box}
TV_weight_app       = {TV_weight_app_box}
rm_weight_mask_thre = {rm_weight_mask_thre_box}
alpha_mask_thre     = {alpha_mask_thre_box}
'''
)
        with st.spinner('Wait for it...'):
            time.sleep(1)
        success = st.success('Done!')
        time.sleep(1)
        success.empty()

