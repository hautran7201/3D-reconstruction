import cv2
import numpy as np
import scipy.signal
import torch
import torch.nn.functional as F
import torchvision.transforms as T
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


def visualize_depth(depth, minmax=None, cmap=cv2.COLORMAP_JET):
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

    x = (x - mi) / (ma - mi + 1e-8)  # normalize to 0~1
    x = (255 * x).astype(np.uint8)
    x_ = Image.fromarray(cv2.applyColorMap(x, cmap))
    x_ = T.ToTensor()(x_)  # (3, H, W)
    return x_, [mi, ma]


def N_to_reso(n_voxels, bbox):
    xyz_min, xyz_max = bbox
    dim = len(xyz_min)
    voxel_size = ((xyz_max - xyz_min).prod() / n_voxels).pow(1 / dim)
    return ((xyz_max - xyz_min) / voxel_size).long().tolist()


def cal_n_samples(reso, step_ratio=0.5):
    return int(np.linalg.norm(reso) / step_ratio)


def findItem(items, target):
    for one in items:
        if one[: len(target)] == target:
            return one
    return None


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
