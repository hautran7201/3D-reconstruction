import torch,cv2
import json
import os
import numpy as np 

from tqdm import tqdm
from torch.utils.data import Dataset
from PIL import Image, ImageEnhance
from torchvision import transforms as T

from .ray_utils import *


class DiffuHumanDataset(Dataset):
    def __init__(self, datadir, split='train', downsample=1.0, is_stack=False, N_vis=-1, tqdm=True, N_imgs=0, indexs=[],
    enhance=None):

        self.w = 800 
        self.h = 800

        self.N_vis = N_vis
        self.root_dir = datadir
        self.split = split
        self.is_stack = is_stack
        self.img_wh = (int(self.w/downsample),int(self.h/downsample))
        self.tqdm = tqdm
        self.N_imgs = N_imgs
        self.indexs = indexs
        self.enhance = enhance
        self.define_transforms()

        self.scene_bbox = torch.tensor([[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]])
        self.blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        self.downsample=downsample
        self.read_meta()
        self.define_proj_mat()

        self.white_bg = True
        self.near_far = [2, 6]
        
        self.center = torch.mean(self.scene_bbox, axis=0).float().view(1, 1, 3)
        self.radius = (self.scene_bbox[1] - self.center).float().view(1, 1, 3)

    def read_depth(self, filename):
        depth = np.array(read_pfm(filename)[0], dtype=np.float32)  # (800, 800)
        return depth
    
    def read_meta(self):

        with open(os.path.join(self.root_dir, f"transforms_{self.split}.json"), 'r') as f:
        #with open(r'/content/drive/MyDrive/THuman2.0/rendered_images/0001/transforms_train.json', 'r') as f:
            self.meta = json.load(f)

        self.w, self.h = self.img_wh
        self.focal = 0.5 * 800 / np.tan(0.5 * self.meta['camera_angle_x'])  # original focal length
        self.cx = (self.meta['cx'] / self.downsample) if 'cx' in self.meta else (self.h / 2)
        self.cy = (self.meta['cy'] / self.downsample) if 'cy' in self.meta else (self.w / 2)
        self.intrinsics_info = np.array([self.focal, self.focal, self.cx, self.cy])

        self.focal *= self.img_wh[0] / 800  # modify focal length to match size self.img_wh


        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = get_ray_directions(self.h, self.w, [self.focal,self.focal])  # (h, w, 3)
        self.directions = self.directions / torch.norm(self.directions, dim=-1, keepdim=True)
        self.intrinsics_matrix = torch.tensor([[self.focal,0,self.w/2],[0,self.focal,self.h/2],[0,0,1]]).float()
        

        # print(self.intrinsics_matrix)
        
        self.image_paths = []
        self.image_fulls = []
        self.poses = []
        self.scale_poses = []
        self.all_rays = []
        self.all_rgbs = []
        self.all_masks = []
        self.all_depth = []

        img_eval_interval = 1 if self.N_vis < 0 else len(self.meta['frames']) // self.N_vis
        idxs = list(range(0, len(self.meta['frames']), img_eval_interval))


        if len(self.indexs) > 0:
            idxs = self.indexs
            self.N_imgs = len(self.indexs)
        elif self.N_imgs > 0 and self.N_imgs < len(idxs):
            idxs = np.random.choice(idxs, self.N_imgs, replace=False)


        if self.tqdm:
            bars = tqdm(idxs, desc=f'Loading data {self.split} ({len(idxs)})')
        else:
            bars = idxs  


        for i in bars:#img_list:#
            frame = self.meta['frames'][i]
            pose = np.array(frame['transform_matrix']) @ self.blender2opencv
            c2w = torch.FloatTensor(pose)
            self.poses += [c2w]
            self.scale_poses += [self.nerf_matrix_to_ngp(c2w)]

            file_path = frame['file_path'].split('.')[-1]
            image_path = self.root_dir + file_path + '.png'
            self.image_paths += [image_path]

            """file_path = frame['file_path'].split('\\')[-1].split('.')[-2]
            image_path = os.path.join(self.root_dir, self.split, file_path+'.png')
            self.image_paths += [image_path]"""
            

            img = Image.open(image_path)
            if self.downsample!=1.0:
                img = img.resize(self.img_wh, Image.LANCZOS)

            if len(img.mode) == 4:
                _, _, _, mask = img.split()
                mask = self.transform(mask)
            else:                
                img_shape = img.size
                mask = torch.ones((1, img_shape[0], img_shape[1]))

            img = self.transform(img)  # (4, h, w)
            img_full = img.permute(1, 2, 0)
            img_full = img_full[:, :, :3] * img_full[:, :, -1:] + (1 - img_full[:, :, -1:])
            self.image_fulls.append(img_full)
            
            img = img.view(4, -1).permute(1, 0)  # (h*w, 4) RGBA                
            mask = mask.view(1, -1).permute(1, 0)  # (h*w, 1) RGBA
            img = img[:, :3] * img[:, -1:] + (1 - img[:, -1:])  # blend A to RGB
            
            rays_o, rays_d = get_rays(self.directions, c2w)  # both (h*w, 3)
            self.all_rays += [torch.cat([rays_o, rays_d], 1)]  # (h*w, 6)
            self.all_masks += [mask]
            self.all_rgbs += [img]

        # self.image_fulls = torch.stack(self.image_fulls, 0)
        self.image_fulls = torch.from_numpy(np.stack(self.image_fulls, axis=0))
        self.scale_poses = np.array(self.scale_poses)

        self.poses = torch.stack(self.poses)
        if not self.is_stack:
            self.all_rays = torch.cat(self.all_rays, 0)  # (len(self.meta['frames])*h*w, 3)
            self.all_rgbs = torch.cat(self.all_rgbs, 0)  # (len(self.meta['frames])*h*w, 3)
            self.all_masks = torch.cat(self.all_masks, 0)  # (len(self.meta['frames]),h,w,3)
            # self.all_depth = torch.cat(self.all_depth, 0)  # (len(self.meta['frames])*h*w, 3)

        else:
            self.all_rays = torch.stack(self.all_rays, 0)  # (len(self.meta['frames]),h*w, 3)
            self.all_rgbs = torch.stack(self.all_rgbs, 0).reshape(-1,*self.img_wh[::-1], 3)  # (len(self.meta['frames]),h,w,3)
            self.all_masks = torch.stack(self.all_masks, 0).reshape(-1,*self.img_wh[::-1])  # (len(self.meta['frames]),h,w,3)


    def nerf_matrix_to_ngp(self, pose, scale=0.33):
        # for the fox dataset, 0.33 scales camera radius to ~ 2
        new_pose = np.array([
            [pose[1, 0], -pose[1, 1], -pose[1, 2], pose[1, 3] * scale],
            [pose[2, 0], -pose[2, 1], -pose[2, 2], pose[2, 3] * scale],
            [pose[0, 0], -pose[0, 1], -pose[0, 2], pose[0, 3] * scale],
            [0, 0, 0, 1],
        ], dtype=np.float32)
        return new_pose

    def define_transforms(self):
        self.transform = T.ToTensor()
        
    def define_proj_mat(self):
        self.proj_mat = self.intrinsics_matrix.unsqueeze(0) @ torch.inverse(self.poses)[:,:3]

    def world2ndc(self,points,lindisp=None):
        device = points.device
        return (points - self.center.to(device)) / self.radius.to(device)
    
    @property
    def fov_x(self):
        return 2. * np.arctan(self.w / (2. * self.intrinsics_info[0]))

    @property
    def fov_y(self):
        return 2. * np.arctan(self.h / (2. * self.intrinsics_info[1]))
        
    def __len__(self):
        return len(self.all_rgbs)

    def __getitem__(self, idx):
        if self.split == 'train':  # use data in the buffers
            sample = {'rays': self.all_rays[idx],
                      'rgbs': self.all_rgbs[idx]}

        else:  # create data for each image separately

            img = self.all_rgbs[idx]
            rays = self.all_rays[idx]
            mask = self.all_masks[idx] # for quantity evaluation

            sample = {'rays': rays,
                      'rgbs': img,
                      'mask': mask}
        
        return sample

    def get_img(self, id_image=None):
        if id_image in self.indexs:
            idx = self.indexs.index(id_image)
        else:
            idx = np.random.randint(self.N_imgs)

        return {
            'image': self.image_fulls[idx],
            'intrinsic': self.intrinsics_info,
            'pose': self.poses[idx]
        }
