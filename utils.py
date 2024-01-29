import cv2
import numpy as np
import scipy.signal
import torch
import torch.nn.functional as F
import torchvision.transforms as T

from packaging import version as pver
from PIL import Image

mse2psnr = lambda x: -10.0 * torch.log(x) / torch.log(torch.Tensor([10.0]))

def get_freq_reg_mask(pos_enc_lengths, current_iter, total_reg_iter, ratio, max_visible=None, type='submission', device='cpu'):

    freq_maskes = []
    for pos_enc_length in pos_enc_lengths:
        if max_visible is None:
          # default FreeNeRF
          dv = 4
          if current_iter < total_reg_iter:
            freq_mask = torch.zeros(pos_enc_length).to(device)  # all invisible
            pos_enc_length = pos_enc_length * ratio
            ptr = pos_enc_length / dv * current_iter / total_reg_iter + 1 
            ptr = ptr if ptr < pos_enc_length / dv else pos_enc_length / dv
            int_ptr = int(ptr)
            freq_mask[: int_ptr * dv] = 1.0  # assign the integer part
            freq_mask[int_ptr * dv : int_ptr * dv + dv] = (ptr - int_ptr)  # assign the fractional part
            return torch.clamp(freq_mask, 1e-8, 1 - 1e-8)
          else:
            return torch.ones(pos_enc_length).to(device)
        else:
          # For the ablation study that controls the maximum visible range of frequency spectrum
          freq_mask = torch.zeros(pos_enc_length).to(device)
          freq_mask[: int(pos_enc_length * max_visible)] = 1.0
          freq_maskes.append(freq_mask)

    return freq_maskes

def get_free_mask(pos_bl=[0], view_bl=[0], fea_bl=[0], den_bl=[], app_bl=[], step=-1, total_step=1, ratio=1, using_decomp_mask=True, max_visible=None, device='cpu'):
  pos_mask = None
  view_mask = None
  fea_mask = None
  den_mask = None
  app_mask = None

  if pos_bl[0] > 0:
      pos_mask = get_freq_reg_mask(pos_bl, step, total_step, ratio=ratio, max_visible=max_visible, type='submission', device=device)[0]
  if view_bl[0] > 0:
      view_mask = get_freq_reg_mask(view_bl, step, total_step, ratio=ratio, max_visible=max_visible, type='submission', device=device)[0]
  if fea_bl[0] > 0:
      fea_mask = get_freq_reg_mask(fea_bl, step, total_step, ratio=ratio, max_visible=max_visible, type='submission', device=device)[0]
  if using_decomp_mask:
      if len(den_bl) > 0:
          den_mask = get_freq_reg_mask(den_bl, step, total_step, ratio=ratio, max_visible=max_visible, type='submission', device=device)
      if len(app_bl) > 0:
          app_mask = get_freq_reg_mask(app_bl, step, total_step, ratio=ratio, max_visible=max_visible, type='submission', device=device)
  else: 
      den_mask = None
      app_mask = None

  return {
    'encoding': {
      'pos': pos_mask,
      'view': view_mask,
      'fea': fea_mask
    },
    'decomp': {
      'den': den_mask,
      'app': app_mask
    }
  }

def lossfun_occ_reg(rgb, density, reg_range=10, wb_prior=False, wb_range=20):
    '''
    Computes the occulusion regularization loss.

    Args:
        rgb (torch.Tensor): The RGB rays/images.
        density (torch.Tensor): The current density map estimate.
        reg_range (int): The number of initial intervals to include in the regularization mask.
        wb_prior (bool): If True, a prior based on the assumption of white or black backgrounds is used.
        wb_range (int): The range of RGB values considered to be a white or black background.

    Returns:
        float: The mean occlusion loss within the specified regularization range and white/black background region.
    '''
    # Compute the mean RGB value over the last dimension
    rgb_mean = rgb.mean(dim=-1)
    
    # Compute a mask for the white/black background region if using a prior
    if wb_prior:
        white_mask = (rgb_mean > 0.99).float()  # A naive way to locate white background
        black_mask = (rgb_mean < 0.01).float()  # A naive way to locate black background
        rgb_mask = (white_mask + black_mask)  # White or black background
        rgb_mask[:, wb_range:] = 0  # White or black background range
    else:
        rgb_mask = torch.zeros_like(rgb_mean)
    
    # Create a mask for the general regularization region
    if reg_range > 0:
        rgb_mask[:, :reg_range] = 1  # Penalize the points in reg_range close to the camera
    
    # Compute the density-weighted loss within the regularization and white/black background mask
    return (density * rgb_mask).mean()

def visualize_depth_numpy(depth, minmax=None, cmap=cv2.COLORMAP_JET):
    """
    depth: (H, W)
    """

    x = np.nan_to_num(depth)  # change nan to 0
    if minmax is None:
        mi = np.min(x[x > 0])  # get minimum positive depth (ignore background)
        ma = np.max(x)
    else:
        mi, ma = minmax

    x = (x - mi) / (ma - mi + 1e-8)  # normalize to 0~1
    x = (255 * x).astype(np.uint8)
    x_ = cv2.applyColorMap(x, cmap)
    return x_, [mi, ma]


def init_log(log, keys):
    for key in keys:
        log[key] = torch.tensor([0.0], dtype=float)
    return log


def visualize_depth(depth, minmax=None, return_normalize=True, cmap=cv2.COLORMAP_JET):
    """
    depth: (H, W)
    """
    if type(depth) is not np.ndarray:
        depth = depth.cpu().numpy()

    x = np.nan_to_num(depth)  # change nan to 0
    if minmax is None:
        mi = np.min(x[x > 0])  # get minimum positive depth (ignore background)
        ma = np.max(x)
    else:
        mi, ma = minmax

    if return_normalize:
        x = (x - mi) / (ma - mi + 1e-8)  # normalize to 0~1
    else:
        x = (x - mi) / (ma - mi + 1e-8)  # normalize to 0~1
        x = (255 * x).astype(np.uint8)
        print('x_ = Image.fromarray(cv2.applyColorMap(x, cmap))')
        x_ = Image.fromarray(cv2.applyColorMap(x, cmap))
        print(x_.shape)
    
    return x, [mi, ma]


def N_to_reso(n_voxels, bbox):
    xyz_min, xyz_max = bbox
    dim = len(xyz_min)
    voxel_size = ((xyz_max - xyz_min).prod() / n_voxels).pow(1 / dim)
    return ((xyz_max - xyz_min) / voxel_size).long().tolist()


def cal_n_samples(reso, step_ratio=0.5):
    return int(np.linalg.norm(reso) / step_ratio)


__LPIPS__ = {}


def init_lpips(net_name, device):
    assert net_name in ["alex", "vgg"]
    import lpips

    print(f"init_lpips: lpips_{net_name}")
    return lpips.LPIPS(net=net_name, version="0.1").eval().to(device)


def rgb_lpips(np_gt, np_im, net_name, device):
    if net_name not in __LPIPS__:
        __LPIPS__[net_name] = init_lpips(net_name, device)
    gt = torch.from_numpy(np_gt).permute([2, 0, 1]).contiguous().to(device)
    im = torch.from_numpy(np_im).permute([2, 0, 1]).contiguous().to(device)
    return __LPIPS__[net_name](gt, im, normalize=True).item()


def findItem(items, target):
    for one in items:
        if one[: len(target)] == target:
            return one
    return None


""" Evaluation metrics (ssim, lpips)
"""


def rgb_ssim(
    img0,
    img1,
    max_val,
    filter_size=11,
    filter_sigma=1.5,
    k1=0.01,
    k2=0.03,
    return_map=False,
):
    # Modified from https://github.com/google/mipnerf/blob/16e73dfdb52044dcceb47cda5243a686391a6e0f/internal/math.py#L58
    assert len(img0.shape) == 3
    assert img0.shape[-1] == 3
    assert img0.shape == img1.shape

    # Construct a 1D Gaussian blur filter.
    hw = filter_size // 2
    shift = (2 * hw - filter_size + 1) / 2
    f_i = ((np.arange(filter_size) - hw + shift) / filter_sigma) ** 2
    filt = np.exp(-0.5 * f_i)
    filt /= np.sum(filt)

    # Blur in x and y (faster than the 2D convolution).
    def convolve2d(z, f):
        return scipy.signal.convolve2d(z, f, mode="valid")

    filt_fn = lambda z: np.stack(
        [
            convolve2d(convolve2d(z[..., i], filt[:, None]), filt[None, :])
            for i in range(z.shape[-1])
        ],
        -1,
    )
    mu0 = filt_fn(img0)
    mu1 = filt_fn(img1)
    mu00 = mu0 * mu0
    mu11 = mu1 * mu1
    mu01 = mu0 * mu1
    sigma00 = filt_fn(img0**2) - mu00
    sigma11 = filt_fn(img1**2) - mu11
    sigma01 = filt_fn(img0 * img1) - mu01

    # Clip the variances and covariances to valid values.
    # Variance must be non-negative:
    sigma00 = np.maximum(0.0, sigma00)
    sigma11 = np.maximum(0.0, sigma11)
    sigma01 = np.sign(sigma01) * np.minimum(
        np.sqrt(sigma00 * sigma11), np.abs(sigma01)
    )
    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2
    numer = (2 * mu01 + c1) * (2 * sigma01 + c2)
    denom = (mu00 + mu11 + c1) * (sigma00 + sigma11 + c2)
    ssim_map = numer / denom
    ssim = np.mean(ssim_map)
    return ssim_map if return_map else ssim


import torch.nn as nn


class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, : h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, : w_x - 1]), 2).sum()
        return (
            self.TVLoss_weight
            * 2
            * (h_tv / count_h + w_tv / count_w)
            / batch_size
        )

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]


import plyfile
import skimage.measure


def convert_sdf_samples_to_ply(
    pytorch_3d_sdf_tensor,
    ply_filename_out,
    bbox,
    level=0.5,
    offset=None,
    scale=None,
):
    """
    Convert sdf samples to .ply

    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to

    This function adapted from: https://github.com/RobotLocomotion/spartan
    """

    numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.numpy()
    voxel_size = list(
        (bbox[1] - bbox[0]) / np.array(pytorch_3d_sdf_tensor.shape)
    )

    verts, faces, normals, values = skimage.measure.marching_cubes(
        numpy_3d_sdf_tensor, level=level, spacing=voxel_size
    )
    faces = faces[..., ::-1]  # inverse face orientation

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = bbox[0, 0] + verts[:, 0]
    mesh_points[:, 1] = bbox[0, 1] + verts[:, 1]
    mesh_points[:, 2] = bbox[0, 2] + verts[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    # try writing to the ply file

    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    verts_tuple = np.zeros(
        (num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")]
    )

    for i in range(0, num_verts):
        verts_tuple[i] = tuple(mesh_points[i, :])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(
        faces_building, dtype=[("vertex_indices", "i4", (3,))]
    )

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    print("saving mesh to %s" % (ply_filename_out))
    ply_data.write(ply_filename_out)

@torch.cuda.amp.autocast(enabled=False)
def get_rays(poses, intrinsics, H, W, N=-1, error_map=None):
    ''' get rays
    Args:
        poses: [B, 4, 4], cam2world
        intrinsics: [4]
        H, W, N: int
        error_map: [B, 128 * 128], sample probability based on training error
    Returns:
        rays_o, rays_d: [B, N, 3]
        inds: [B, N]
    '''

    device = poses.device
    B = poses.shape[0]
    fx, fy, cx, cy = intrinsics

    i, j = custom_meshgrid(torch.linspace(0, W - 1, W, device=device), torch.linspace(0, H - 1, H, device=device))
    i = i.t().reshape([1, H*W]).expand([B, H*W]) + 0.5
    j = j.t().reshape([1, H*W]).expand([B, H*W]) + 0.5

    results = {}

    if N > 0:
        N = min(N, H*W)

        if error_map is None:
            inds = torch.randint(0, H*W, size=[N], device=device) # may duplicate
            inds = inds.expand([B, N])
        else:

            # weighted sample on a low-reso grid
            inds_coarse = torch.multinomial(error_map.to(device), N, replacement=False) # [B, N], but in [0, 128*128)

            # map to the original resolution with random perturb.
            inds_x, inds_y = inds_coarse // 128, inds_coarse % 128 # `//` will throw a warning in torch 1.10... anyway.
            sx, sy = H / 128, W / 128
            inds_x = (inds_x * sx + torch.rand(B, N, device=device) * sx).long().clamp(max=H - 1)
            inds_y = (inds_y * sy + torch.rand(B, N, device=device) * sy).long().clamp(max=W - 1)
            inds = inds_x * W + inds_y

            results['inds_coarse'] = inds_coarse # need this when updating error_map

        i = torch.gather(i, -1, inds)
        j = torch.gather(j, -1, inds)

    else:
        inds = torch.arange(H*W, device=device).expand([B, H*W])

    zs = torch.ones_like(i)
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    directions = torch.stack((xs, ys, zs), dim=-1)
    directions = directions / torch.norm(directions, dim=-1, keepdim=True)
    rays_d = directions @ poses[:, :3, :3].transpose(-1, -2) # (B, N, 3)

    rays_o = poses[..., :3, 3] # [B, 3]
    rays_o = rays_o[..., None, :].expand_as(rays_d) # [B, N, 3]

    results['rays_o'] = rays_o
    results['rays_d'] = rays_d
    results['rays_d_cam'] = directions
    results['rays_d_cam_z'] = directions[..., -1] # Useful because it's the conversion factor to go from spherical to planar depths
    results['inds'] = inds

    return results

def custom_meshgrid(*args):
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if pver.parse(torch.__version__) < pver.parse('1.10'):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing='ij')

@torch.jit.script
def srgb_to_linear(x):
    return torch.where(x < 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
