from .llff import LLFFDataset
from .blender import BlenderDataset
from .nsvf import NSVF
from .tankstemple import TanksTempleDataset
from .human import HumanDataset
from .test_diffusion import DiffuHumanDataset



dataset_dict = {'blender': BlenderDataset,
                'llff':LLFFDataset,
                'tankstemple':TanksTempleDataset,
                'nsvf':NSVF,
                'human':HumanDataset,
                'test_diffusion': DiffuHumanDataset}