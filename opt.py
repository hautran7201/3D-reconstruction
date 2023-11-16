import configargparse

def config_parser(cmd=None):
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    parser.add_argument("--expname", type=str,
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./log',
                        help='where to store ckpts and logs')
    parser.add_argument("--add_timestamp", type=int, default=0,
                        help='add timestamp to dir')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern',
                        help='input data directory')
    parser.add_argument("--progress_refresh_rate", type=int, default=10,
                        help='how many iterations to show psnrs or iters')

    parser.add_argument('--with_depth', action='store_true')
    parser.add_argument('--downsample_train', type=float, default=1.0)
    parser.add_argument('--downsample_test', type=float, default=1.0)

    parser.add_argument('--model_name', type=str, default='TensorVMSplit',
                        choices=['TensorVMSplit', 'TensorCP'])

    # loader options
    parser.add_argument("--train_batch_size", type=int, default=4096)
    parser.add_argument("--entropy_batch_size", type=int, default=4096)
    parser.add_argument("--n_iters", type=int, default=30000)

    parser.add_argument('--dataset_name', type=str, default='blender',
                        choices=['blender', 'llff', 'nsvf', 'dtu','tankstemple', 'own_data'])

    # Infor nerf 
    parser.add_argument("--info_nerf", action='store_true', 
                        help='apply info nerf')

    # training options
    # optimizing 
    parser.add_argument("--smoothing_lambda", type=float, default=1, 
                        help='lambda for smoothing loss')
    parser.add_argument("--smoothing_activation", type=str, default='norm', 
                        help='how to make alpha to the distribution')
    parser.add_argument("--smoothing_step_size", type=int, default='5000',
                        help='reducing smoothing every')
    parser.add_argument("--smoothing_rate", type=float, default=1,
                        help='reducing smoothing rate')
    parser.add_argument("--smoothing_end_iter", type=int, default=None,
                        help='when smoothing will be end')
    parser.add_argument("--lr_init", type=float, default=0.02,
                        help='learning rate')    
    parser.add_argument("--lr_basis", type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument("--lr_decay_iters", type=int, default=-1,
                        help = 'number of iterations the lr will decay to the target ratio; -1 will set it to n_iters')
    parser.add_argument("--lr_decay_target_ratio", type=float, default=0.1,
                        help='the target decay ratio; after decay_iters inital lr decays to lr*ratio')
    parser.add_argument("--lr_upsample_reset", type=int, default=1,
                        help='reset lr to inital after upsampling')    
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops')    
                                                 
 
    # loss
    # L1
    parser.add_argument("--L1_reg", action='store_true',
                        help='using L1 reg')
    parser.add_argument("--L1_weight_inital", type=float, default=0.0,
                        help='loss weight')
    parser.add_argument("--L1_weight_rest", type=float, default=0,
                        help='loss weight')
    # Ortho
    parser.add_argument("--ortho_reg", action='store_true',
                        help='using ortho reg')
    parser.add_argument("--Ortho_weight", type=float, default=0.0,
                        help='loss weight')
    # Tv
    parser.add_argument("--tv_reg_density", action='store_true',
                        help='using tv reg density')
    parser.add_argument("--TV_weight_density", type=float, default=0.0,
                        help='loss weight')
    parser.add_argument("--tv_reg_app", action='store_true',
                        help='using tv reg app')                        
    parser.add_argument("--TV_weight_app", type=float, default=0.0,
                        help='loss weight')
    # Occ
    parser.add_argument("--occ_reg", action='store_true',
                        help='using occlusion reg')
    parser.add_argument("--occ_reg_loss_mult", type=float, default=0.0,
                        help='loss occlusion')
    parser.add_argument("--occ_reg_range", type=int, default=0,
                        help='reg range occlusion')
    parser.add_argument("--occ_wb_range", type=int, default=0,
                        help='wb range occlusion')                        
    parser.add_argument("--occ_wb_prior", type=bool, default=False,
                        help='prior occlusion')
    # Entropy 
    parser.add_argument("--entropy", action='store_true',
                        help='using entropy ray loss')
    parser.add_argument("--entropy_type", type=str, default='log2', choices=['log2', '1-p'],
                        help='choosing type of entropy')                        
    parser.add_argument("--entropy_acc_threshold", type=float, default=0.1,
                        help='threshold for acc masking')
    parser.add_argument("--computing_entropy_all", action='store_true',
                        help='computing entropy for both seen and unseen')
    parser.add_argument("--smoothing", action='store_true',
                        help='using information gain reduction loss')                        
    parser.add_argument("--entropy_ignore_smoothing", action='store_true',
                        help='ignoring entropy for ray for smoothing')
    parser.add_argument("--entropy_log_scaling", action='store_true',
                        help='using log scaling for entropy loss')
    parser.add_argument("--N_entropy", type=int, default=100,
                        help='number of entropy ray')
    parser.add_argument("--entropy_end_iter", type=int, default=None,
                        help='end iteratio of entropy')                        
    parser.add_argument("--no_batching", action='store_true', 
                        help='only take random rays from 1 image at a time')

    #lambda
    parser.add_argument("--entropy_ray_lambda", type=float, default=1,
                        help='entropy lambda for ray entropy loss')
    parser.add_argument("--entropy_ray_zvals_lambda", type=float, default=1,
                        help='entropy lambda for ray zvals entropy loss')                        

    # choosing between rotating camera pose & near pixel
    parser.add_argument("--smooth_sampling_method", type=str, default='near_pose', 
        help='how to sample the near rays, near_pose: modifying camera pose, near_pixel: sample near pixel', 
                    choices=['near_pose', 'near_pixel'])                        
    # Sampling by rotating camera pose
    parser.add_argument("--near_c2w_type", type=str, default='rot_from_origin', 
                        help='random augmentation method')
    parser.add_argument("--near_c2w_rot", type=float, default=5, 
                        help='random augmentation rotate: degree')
    parser.add_argument("--near_c2w_trans", type=float, default=0.1, 
                        help='random augmentation translation')                        

    # Regularization
    parser.add_argument("--free_reg", action='store_true',
                        help='using entropy ray loss')
    parser.add_argument("--freq_reg_ratio", type=float, default=1,
                        help='encoding reg ratio')
    parser.add_argument("--max_vis_freq_ratio", type=float, default=0.0,
                        help='encoding reg')


    # model
    # volume options
    parser.add_argument("--n_lamb_sigma", type=int, action="append")
    parser.add_argument("--n_lamb_sh", type=int, action="append")
    parser.add_argument("--data_dim_color", type=int, default=27)

    parser.add_argument("--rm_weight_mask_thre", type=float, default=0.0001,
                        help='mask points in ray marching')
    parser.add_argument("--alpha_mask_thre", type=float, default=0.0001,
                        help='threshold for creating alpha mask volume')
    parser.add_argument("--distance_scale", type=float, default=25,
                        help='scaling sampling distance for computation')
    parser.add_argument("--density_shift", type=float, default=-10,
                        help='shift density in softplus; making density = 0  when feature == 0')

    # nerf_v3
    parser.add_argument("--netwidth", type=int, default=256,
                        help='width network')
    parser.add_argument("--netdepth", type=int, default=22,
                        help='depth network')  
    parser.add_argument('--body_arch', type=str, default='mlp',
                        choices=['mlp', 'resmlp'])
    parser.add_argument('--resmlp_act', type=str, default='relu',
                        choices=['relu', 'lrelu'])                        
    parser.add_argument("--linear_tail", action="store_true",
                        help='tail network')
    parser.add_argument("--use_residual", action="store_true",
                        help='residual at body input')
    parser.add_argument("--trial", action="store_true",
                        help='use residual')      
    parser.add_argument('--layerwise_netwidths', type=str, default='')                                                                                                       
    parser.add_argument('--dropout', type=float, default=1)

    # alpha_mask 
    parser.add_argument("--rayMarch_weight_thres", type=float, default=0.0001,
                        help='way march weigth threshold')
                                  
    # network decoder
    parser.add_argument("--shadingMode", type=str, default="MLP_PE",
                        help='which shading mode to use')
    parser.add_argument("--pos_pe", type=int, default=6,
                        help='number of pe for pos')
    parser.add_argument("--view_pe", type=int, default=6,
                        help='number of pe for view')
    parser.add_argument("--fea_pe", type=int, default=6,
                        help='number of pe for features')
    parser.add_argument("--featureC", type=int, default=128,
                        help='hidden feature channel in MLP')

    parser.add_argument("--ckpt", type=str, default=None,
                        help='specific weights npy file to reload for coarse network')
                        
    parser.add_argument("--render_only", type=int, default=0)
    parser.add_argument("--render_test", type=int, default=0)
    parser.add_argument("--render_train", type=int, default=0)
    parser.add_argument("--render_path", type=int, default=0)
    parser.add_argument("--export_mesh", type=int, default=0)

    # rendering options
    parser.add_argument('--lindisp', default=False, action="store_true",
                        help='use disparity depth sampling')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--accumulate_decay", type=float, default=0.998)
    parser.add_argument("--fea2denseAct", type=str, default='softplus')
    parser.add_argument('--ndc_ray', type=int, default=0)
    parser.add_argument('--nSamples', type=int, default=1e6,
                        help='sample point each ray, pass 1e6 if automatic adjust')
    parser.add_argument('--step_ratio',type=float,default=0.5)


    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true',
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument('--N_voxel_init',
                        type=int,
                        default=100**3)
    parser.add_argument('--N_voxel_final',
                        type=int,
                        default=300**3)
    parser.add_argument("--upsamp_list", type=int, action="append")
    parser.add_argument("--update_AlphaMask_list", type=int, action="append")
    parser.add_argument("--reset_allray", type=int, action="append")

    parser.add_argument('--idx_view', type=int, default=0)


    # logging/saving options
    parser.add_argument("--N_vis", type=int, default=5,
                        help='N images to vis')
    parser.add_argument("--vis_every", type=int, default=10000,
                        help='frequency of visualize the image')  
    parser.add_argument("--train_vis_every", type=int, default=1000,
                        help='visualize the training image every')    
    parser.add_argument("--overwrt", action="store_true",
                        help='overwrite checkpoint')                            

    if cmd is not None:
        return parser.parse_args(cmd)
    else:
        return parser.parse_args()