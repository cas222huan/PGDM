import glob
import h5py
import numpy as np
from natsort import natsorted
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms, InterpolationMode
from torchvision.transforms import functional as F
from .utils import calculate_spectral_indices

lulc_encode = -np.ones(201, dtype=np.int32)
valid_loc = [0, 20, 30, 40, 50, 60, 70, 80, 90, 100, 111, 112, 113, 114, 115, 116, 121, 122, 123, 124, 125, 126, 200]
lulc_encode[valid_loc] = np.arange(len(valid_loc))

# minmax normalization to [vmin, vmax]
def normalize_data(x, xmin, xmax, vmin=-1, vmax=1):
        return (x - xmin) / (xmax - xmin) * (vmax - vmin) + vmin

def denormalize_data(x_norm, xmin, xmax, vmin=-1, vmax=1):
    return (x_norm - vmin) / (vmax - vmin) * (xmax - xmin) + xmin

def get_all_files(folder, indices=None, file_type='h5'):
    files = np.array(natsorted(glob.glob(folder + f'/*.{file_type}')))
    if indices is not None:
        indices = np.array(indices)
        files = files[indices]
    return files

class MyDataset(Dataset):
    def __init__(self, file_path_all, mode:str='train', flip_during_train:bool=True, drop_prob:float=0.2,
                 hr_size:int=160, scale_factor:int=10, itp_mode:str='bilinear', fixed_drop_channels:list=None,
                 normalize_params={'lst_min':180., 'lst_max':360., 
                                   'ref_min':0., 'ref_max':1.,
                                   'ndxi_min':-1., 'ndxi_max':1.,
                                   'dem_min':-400., 'dem_max':8800.,
                                   'v_min':-1., 'v_max':1.}):
        assert scale_factor in [10, 20], "scale_factor must be 10 or 20"
        assert hr_size % scale_factor == 0, "hr_size must be divisible by scale_factor"
        assert itp_mode in ['bilinear', 'nearest'], "itp_mode must be 'bilinear' or 'nearest'"
        self.files = file_path_all
        self.mode = mode
        self.flip_during_train = flip_during_train  # whether to flip the images during training
        self.drop_prob = drop_prob  # probability to drop each guidance channel during training
        self.scale_factor = scale_factor
        self.normalize_params = normalize_params
        self.fixed_drop_channels = fixed_drop_channels
        self.lst_lr_order = scale_factor // 10
        itp_way = InterpolationMode.BILINEAR if itp_mode=='bilinear' else InterpolationMode.NEAREST
        self.lr_2_hr = transforms.Resize((hr_size, hr_size), interpolation=itp_way)
    
    def __len__(self):
        return len(self.files)
    
    def normalize(self, x, data_type):
        assert data_type in ['lst', 'ref', 'ndxi', 'dem'], "data_type must be one of ['lst', 'ref', 'ndxi', 'dem']"
        x_norm = normalize_data(x, self.normalize_params[f'{data_type}_min'], self.normalize_params[f'{data_type}_max'],
                               self.normalize_params['v_min'], self.normalize_params['v_max'])
        return x_norm

    def denormalize(self, x_norm, data_type):
        assert data_type in ['lst', 'ref', 'ndxi', 'dem'], "data_type must be one of ['lst', 'ref', 'ndxi', 'dem']"
        x = denormalize_data(x_norm, self.normalize_params[f'{data_type}_min'], self.normalize_params[f'{data_type}_max'],
                             self.normalize_params['v_min'], self.normalize_params['v_max'])
        return x

    def _read_h5_file(self, file_path):
        ds = h5py.File(file_path, 'r')
        # shape: (C, H, W)
        lst_hr, lst_lr, ref, dem = ds['lst_hr'][:][None,...], ds[f'lst_lr_{self.lst_lr_order}'][:][None,...], ds['ref_hr'][:], ds['dem_hr'][:][None,...]
        try:
            lulc = ds['lulc_hr'][:][None,...].astype(np.int32)
            lulc = lulc_encode[lulc] # encode to 0~22
        except KeyError:
            lulc = np.zeros_like(dem, dtype=np.int32)
        ds.close()

        ndvi = calculate_spectral_indices(ref[3], ref[2])  # NIR and R
        ndwi = calculate_spectral_indices(ref[1], ref[3])  # G and NIR
        ndmi = calculate_spectral_indices(ref[3], ref[4])  # NIR and SWIR1
        ndxi = np.stack((ndvi, ndwi, ndmi), axis=0)

        return lst_hr, lst_lr, ref, ndxi, dem, lulc
    
    def _normalize_all(self, lst_hr, lst_lr, ref, ndxi, dem):
        lst_hr = self.normalize(lst_hr, 'lst')
        lst_lr = self.normalize(lst_lr, 'lst')
        ref = self.normalize(ref, 'ref')
        ndxi = self.normalize(ndxi, 'ndxi')
        dem = self.normalize(dem, 'dem')
        return lst_hr, lst_lr, ref, ndxi, dem
    
    def _to_tensors(self, array_list, dtype_list):
        tensor_list = []
        for arr, dt in zip(array_list, dtype_list):
            tensor_list.append(torch.tensor(arr, dtype=dt))
        return tensor_list
    
    def __getitem__(self, idx):
        lst_hr, lst_lr, ref, ndxi, dem, lulc = self._read_h5_file(self.files[idx])

        # data augmentation (randomly drop)
        num_gui = len(ref) + len(ndxi) + len(dem) + len(lulc) # 6+3+1+1
        mask = np.ones(num_gui, dtype=np.float32)
        if self.mode == 'train':
            # randomly drop channel during training
            for i in range(num_gui):
                if np.random.rand() < self.drop_prob:
                    mask[i] = 0.0
        else: # val or test
            # fixed drop during testing
            if self.fixed_drop_channels is not None:
                mask[self.fixed_drop_channels] = 0.0

        # normalize data (min-max to [-1, 1])
        lst_hr, lst_lr, ref, ndxi, dem = self._normalize_all(lst_hr, lst_lr, ref, ndxi, dem)
        # convert to torch tensors
        lst_hr, lst_lr, ref, ndxi, dem, lulc, mask = self._to_tensors([lst_hr, lst_lr, ref, ndxi, dem, lulc, mask], 
                                                                      [torch.float32, torch.float32, torch.float32, torch.float32, torch.float32, torch.int32, torch.float32])
        lst_lr_itp = self.lr_2_hr(lst_lr)

        # data augmentation (flip)
        if self.mode == 'train' and self.flip_during_train:
            if np.random.rand() < 0.5:
                lst_hr, lst_lr, lst_lr_itp = F.hflip(lst_hr), F.hflip(lst_lr), F.hflip(lst_lr_itp)
                ref, ndxi, dem, lulc = F.hflip(ref), F.hflip(ndxi), F.hflip(dem), F.hflip(lulc)
            if np.random.rand() < 0.5:
                lst_hr, lst_lr, lst_lr_itp = F.vflip(lst_hr), F.vflip(lst_lr), F.vflip(lst_lr_itp)
                ref, ndxi, dem, lulc = F.vflip(ref), F.vflip(ndxi), F.vflip(dem), F.vflip(lulc)

            # angle = int(np.random.choice([0, 90, 180, 270]))
            # if angle != 0:
            #     lst_hr, lst_lr, lst_lr_itp = F.rotate(lst_hr, angle), F.rotate(lst_lr, angle), F.rotate(lst_lr_itp, angle)
            #     ref, ndxi, dem, lulc = F.rotate(ref, angle), F.rotate(ndxi, angle), F.rotate(dem, angle), F.rotate(lulc, angle)

        return lst_hr, lst_lr, lst_lr_itp, ref, ndxi, dem, lulc, mask



if __name__ == "__main__":
    # randomly split Landsat_CN20 dataset into train/val/test sets
    # in root path: python -m lib.dataset
    folder = r'C:\dataset_Landsat_CN20'
    num_files = len(glob.glob(folder + f'/*.h5'))

    np.random.seed(0)
    indices = np.arange(num_files)
    np.random.shuffle(indices)

    train_ratio, val_ratio, test_ratio = 0.7, 0.15, 0.15
    train_end = int(num_files * train_ratio)
    val_end = train_end + int(num_files * val_ratio)

    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    np.savetxt('dataset/train.txt', train_indices, fmt='%d')
    np.savetxt('dataset/val.txt', val_indices, fmt='%d')
    np.savetxt('dataset/test.txt', test_indices, fmt='%d')

