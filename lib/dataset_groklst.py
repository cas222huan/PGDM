import glob
import os
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms, InterpolationMode

# datafield_name = ['dem','deepblue','blue','green','red','vre','nir','ndmvi','ndvi','ndwi']

stats_dem = {
    "dem": {"mean": 2569.38, "std": 1254.35, "min": 905.42, "max": 5082.07}}

stats_ref = {
    "deepblue": {"mean": 0.17, "std": 0.05, "min": 0.00, "max": 0.71},
    "blue": {"mean": 0.16, "std": 0.04, "min": 0.00, "max": 0.50},
    "green": {"mean": 0.15, "std": 0.04, "min": 0.00, "max": 0.37},
    "red": {"mean": 0.17, "std": 0.05, "min": 0.00, "max": 0.40},
    "vre": {"mean": 0.20, "std": 0.06, "min": 0.00, "max": 0.61},
    "nir": {"mean": 0.22, "std": 0.06, "min": 0.00, "max": 0.60}
}

stats_ndxi = {
    "ndmvi": {"mean": 0.19, "std": 0.13, "min": -0.98, "max": 0.99},
    "ndvi": {"mean": 0.14, "std": 0.10, "min": -0.52, "max": 0.71},
    "ndwi": {"mean": -0.19, "std": 0.12, "min": -0.57, "max": 0.79}
}

stats_lst = {
    "30m": {"mean": 282.51, "std": 14.75, "min": 242.13, "max": 330.92},
    "60m": {"mean": 282.51, "std": 14.73, "min": 242.50, "max": 322.33},
    "120m": {"mean": 282.51, "std": 14.68, "min": 242.72, "max": 321.21},
    "240m": {"mean": 282.51, "std": 14.59, "min": 243.53, "max": 320.85}
}

class GrokLST_Dataset(Dataset):
    def __init__(self, root_dir, txt_file, zoom:int=8, hr_size:int=512):
        assert zoom in [2, 4, 8], 'zoom must be in [2, 4, 8]'
        self.low_resolution = f'{30*zoom}m'
        lst_hr_dir = os.path.join(root_dir, '30m/lst')
        gui_hr_dir = os.path.join(root_dir, '30m/guidance')
        lst_lr_dir = os.path.join(root_dir, f'{self.low_resolution}/lst')
        index = np.loadtxt(os.path.join(root_dir, txt_file)).astype(int)

        self.img_lst_hr = np.array(sorted(glob.glob(os.path.join(lst_hr_dir, '*.mat'))))[index]
        self.img_gui_hr = np.array(sorted(glob.glob(os.path.join(gui_hr_dir, '*.mat'))))[index]
        self.img_lst_lr = np.array(sorted(glob.glob(os.path.join(lst_lr_dir, '*.mat'))))[index]
        assert len(self.img_lst_hr) == len(self.img_gui_hr) == len(self.img_lst_lr), 'Datasize does not match.'
        
        self.lr_2_hr = transforms.Resize((hr_size, hr_size), interpolation=InterpolationMode.BICUBIC)

    def __len__(self):
        return len(self.img_lst_hr)
 
    def normalize(self, data, stats, field_name):
        # min-max normalization
        return (data - stats[field_name]['min']) / (stats[field_name]['max'] - stats[field_name]['min']) * 2 - 1

    def denormalize_lst(self, data_norm):
        # min-max denormalization
        return (data_norm + 1) / 2 * (stats_lst['30m']['max'] - stats_lst['30m']['min']) + stats_lst['30m']['min']

    def _to_tensors(self, array_list, dtype_list):
        tensor_list = []
        for arr, dt in zip(array_list, dtype_list):
            tensor_list.append(torch.tensor(arr, dtype=dt))
        return tensor_list
    
    def __getitem__(self, idx):
        lst_hr = sio.loadmat(self.img_lst_hr[idx])['data']
        lst_hr = np.transpose(lst_hr, (2, 0, 1)) # change H,W,1 to 1,H,W
        lst_hr = self.normalize(lst_hr, stats_lst, '30m')
        lst_lr = sio.loadmat(self.img_lst_lr[idx])['data']
        lst_lr = np.transpose(lst_lr, (2, 0, 1)) # change H,W,1 to 1,H,W
        lst_lr = self.normalize(lst_lr, stats_lst, '30m') # using 30m stats to normalize all LST data

        dem_hr = sio.loadmat(self.img_gui_hr[idx])['dem'][None, :, :] # change H,W to 1,H,W
        dem_hr = self.normalize(dem_hr, stats_dem, 'dem')

        ref_hr, ndxi_hr = [], []
        for field_name in stats_ref.keys():
            data_hr_i = sio.loadmat(self.img_gui_hr[idx])[field_name]
            ref_hr.append(self.normalize(data_hr_i, stats_ref, field_name))
        for field_name in stats_ndxi.keys():
            data_hr_i = sio.loadmat(self.img_gui_hr[idx])[field_name]
            ndxi_hr.append(self.normalize(data_hr_i, stats_ndxi, field_name))

        ref_hr = np.stack(ref_hr, axis=0)
        ndxi_hr = np.stack(ndxi_hr, axis=0)

        lst_hr, lst_lr, ref_hr, ndxi_hr, dem_hr = self._to_tensors(
            [lst_hr, lst_lr, ref_hr, ndxi_hr, dem_hr],
            [torch.float32, torch.float32, torch.float32, torch.float32, torch.float32]
        )
        lst_lr_itp = self.lr_2_hr(lst_lr)

        lulc = torch.ones_like(dem_hr, dtype=torch.int32) # dummy lulc, not used in current experiments
        mask = np.ones(11, dtype=np.float32)
        mask[-1] = 0.0 # always drop lulc

        return lst_hr, lst_lr, lst_lr_itp, ref_hr, ndxi_hr, dem_hr, lulc, mask