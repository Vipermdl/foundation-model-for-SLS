import os

os.environ["OMP_NUM_THREADS"] = "4"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "4"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "4"  # export NUMEXPR_NUM_THREADS=6
os.environ["GDAL_NUM_THREADS"] = "4"

import torch
import mmengine
import numpy as np
import os.path as osp
import rasterio as rio
from torch.utils.data import Dataset

def getArr(filename):
    with rio.open(filename) as d:
        data = d.read().astype(np.float32)  
        return data

def getArrSeg(filename, agg_to_water=True):
    # data = None
    with rio.open(filename) as d:
        data = d.read().astype(np.int64) 
        return data

class SWED(Dataset):
    """A dataset class implementing all ben-ge data modalities."""

    def __init__(
        self,
        data_dir=None,
        split="train",
        transforms=None,
    ):
        """Dataset class constructor

        keyword arguments:
        data_dir -- string containing the path to the base directory of ben-ge dataset, default: ben-ge-800 directory

        returns:
        BENGE object
        """
        super().__init__()

        # store some definitions
        self.data_dir = data_dir
        self.transforms = transforms
        
        self.ann_file = f"{self.data_dir}/splits/{split}.txt"

        # read in relevant data files and definitions
        self.name = self.data_dir.split("/")[-1]
        self.meta = self.load_data_list()
        
        self.label_names = [
            'Land', 'Water'
        ]
        self.classes = self.label_names        

    def load_data_list(self):
        data_list = []
        if not osp.isdir(self.ann_file) and self.ann_file:
            assert osp.isfile(self.ann_file), \
                f'Failed to load `ann_file` {self.ann_file}'
            lines = mmengine.list_from_file(self.ann_file)
            for line in lines:
                img_path = osp.join(self.data_dir, line) # Get File path
                data_info = dict(img_path=img_path)
                data_info['seg_map_path'] = osp.join(self.data_dir, line.replace('images', 'labels').replace('image', 'chip')) # Get Class Mask
                data_list.append(data_info)
                
        return data_list

    def __getitem__(self, idx):
        """Return sample `idx` as dictionary from the dataset."""
        sample_info = self.meta[idx]

        sample = {}

        sample["image"] = np.load(sample_info['img_path']).transpose(2, 0, 1).astype(np.float32)
        sample["mask"] = np.load(sample_info['seg_map_path']).squeeze().astype(np.int64)

        sample["mask"][sample["mask"] < 0] = 0
        
        for k, v in sample.items():
            sample[k] = torch.tensor(v, dtype=torch.float)

        sample["mask"] = sample["mask"].to(torch.long)
        
        if self.transforms is not None:
            sample = self.transforms(sample)
        
        sample['img_path'] = sample_info['img_path']

        return sample

    def __len__(self):
        """Return length of this dataset."""
        return len(self.meta)

    



