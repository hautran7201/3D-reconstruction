from .llff import LLFFDataset
from .blender import BlenderDataset
from .nsvf import NSVF
from .tankstemple import TanksTempleDataset
from .human import HumanDataset



dataset_dict = {'blender': BlenderDataset,
                'llff':LLFFDataset,
                'tankstemple':TanksTempleDataset,
                'nsvf':NSVF,
                'human':HumanDataset}