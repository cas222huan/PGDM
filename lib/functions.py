from osgeo import gdal
import numpy as np
import cv2
from sklearn.metrics import root_mean_squared_error as rmse
from skimage.metrics import structural_similarity as ssim

def read_tiff(file_path, scale=None, offset=None, missing_value=-999, return_proj=False):
    ds = gdal.Open(file_path)
    data = ds.ReadAsArray().astype(np.float32)
    data[data == missing_value] = np.nan

    if scale is not None:
        data *= scale
    if offset is not None:
        data += offset

    proj = ds.GetProjection() if return_proj else None
    geotrans = ds.GetGeoTransform() if return_proj else None
    ds = None
    return data, proj, geotrans

# for ASTER data
def get_center_lonlat_from_gdalinfo(file, res):
    info = gdal.Info(file, format='json')
    lonlat_bounds = np.array(info['wgs84Extent']['coordinates'][0])[:4]
    lon_min, lon_max = lonlat_bounds[:, 0].min(), lonlat_bounds[:, 0].max()
    lat_min, lat_max = lonlat_bounds[:, 1].min(), lonlat_bounds[:, 1].max()
    lon_center, lat_center = (lon_min + lon_max) / 2, (lat_min + lat_max) / 2
    lon_center = np.round(lon_center/res)*res
    lat_center = np.round(lat_center/res)*res
    return lon_center, lat_center


# crop the whole image into small patches
def crop_image(image, width=160, step=80):
    if image.ndim==2:
        image = image[np.newaxis, ...]
    _, rows, cols = image.shape

    corner_ul_i = np.arange(0, rows - width + 1, step)
    corner_ul_j = np.arange(0, cols - width + 1, step)
    crops = []
    for i in corner_ul_i:
        for j in corner_ul_j:
            crop = (image[:, i:i+width, j:j+width]).squeeze()
            crops.append(crop)
    return crops

# combine the patches into an entire image
def create_pyramid_weight_mask(patch_size=160, weight_min=0.1):
    half = patch_size // 2
    ramp_up = np.linspace(weight_min, 1, half, endpoint=False)
    ramp_down = np.linspace(1, weight_min, half, endpoint=False)
    ramp = np.concatenate([ramp_up, ramp_down])
    weight_mask = ramp.reshape(-1, 1) * ramp.reshape(1, -1)
    return weight_mask

def combine_patches(patches, original_size=(480,480), patch_size=160, step=80, weight_min=0.1):
    H, W = original_size
    weight_mask = create_pyramid_weight_mask(patch_size=patch_size, weight_min=weight_min)
    combined = np.zeros((H, W))
    weight_accumulator = np.zeros((H, W), dtype=np.float32)

    patch_idx = 0
    for i in range(0, H - patch_size + 1, step):
        for j in range(0, W - patch_size + 1, step):
            if patch_idx >= len(patches):
                break
            
            combined[i:i+patch_size, j:j+patch_size] += patches[patch_idx] * weight_mask
            weight_accumulator[i:i+patch_size, j:j+patch_size] += weight_mask
            patch_idx += 1
    
    result = combined / weight_accumulator
    return result

# calculate LR LST from HR LST
def calculate_lr_lst(hr_lst, lr_size, scale_factor):
    return np.mean(hr_lst.reshape(-1, lr_size, scale_factor, lr_size, scale_factor)**4, axis=(2,4))**0.25

def calculate_accuracy_metrics(lst_hr_true, lst_hr_prd, lr_size, scale_factor, data_range=None):
    # data shape: C x H x W
    if lst_hr_true.ndim == 2:
        lst_hr_true = lst_hr_true[np.newaxis, ...]
        lst_hr_prd = lst_hr_prd[np.newaxis, ...]
    rmse_val = rmse(lst_hr_true.flatten(), lst_hr_prd.flatten())
    bias_val = np.mean(lst_hr_prd - lst_hr_true)
    num_image = lst_hr_true.shape[0]
    ssim_val = 0
    data_range = lst_hr_true.max() - lst_hr_true.min() if data_range is None else data_range
    for i in range(num_image):
        ssim_val_i = ssim(lst_hr_true[i], lst_hr_prd[i], data_range=data_range)
        ssim_val += ssim_val_i
    ssim_val /= num_image

    lst_lr_true = calculate_lr_lst(lst_hr_true, lr_size, scale_factor)
    lst_lr_prd = calculate_lr_lst(lst_hr_prd, lr_size, scale_factor)
    loss_phy = rmse(lst_lr_true.flatten(), lst_lr_prd.flatten())

    return rmse_val, bias_val, ssim_val, loss_phy

def density_scatter(x, y, ax, n_bins=500, gap=10, v_minmax=None, **kwargs):
    counts, xedges, yedges = np.histogram2d(x, y, bins=n_bins)
    ix = np.searchsorted(xedges, x) - 1
    iy = np.searchsorted(yedges, y) - 1
    valid = (ix>=0) & (ix<counts.shape[0]) & (iy>=0) & (iy<counts.shape[1])
    z = np.zeros_like(x, dtype=float)
    z[valid] = counts[ix[valid], iy[valid]]
    z = np.log1p(z)  # log-exp for better visualization

    scatter_obj = ax.scatter(x, y, c=z, **kwargs)
    if v_minmax is None:
        vmin, vmax = np.min([np.min(x), np.min(y)]), np.max([np.max(x), np.max(y)])
        vmin = int(vmin // gap * gap)
        vmax = int(vmax // gap * gap + gap)
    else:
        vmin, vmax = v_minmax
    ax.plot([vmin-1000, vmax+1000], [vmin-1000, vmax+1000], 'k--', linewidth=1)  # 1:1 line
    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)
    ax.set_xticks(np.arange(vmin, vmax+1, gap))
    ax.set_yticks(np.arange(vmin, vmax+1, gap))
    # ax.set_aspect('equal', adjustable='box')
    return scatter_obj


# kernel-driven model
def kernel_driven_model(regression_model, residual_add_way, lst_lr, kernel_hr, factor):
    # lst_lr: H_lr, W_lr; kernel_hr: C, H_hr, W_hr
    assert residual_add_way in ['nearest', 'bilinear', 'cubic']
    itp_methods = {'nearest': cv2.INTER_NEAREST, 'bilinear': cv2.INTER_LINEAR, 'cubic': cv2.INTER_CUBIC}

    H_lr, W_lr = lst_lr.shape
    C, H_hr, W_hr = kernel_hr.shape
    assert H_lr * factor == H_hr and W_lr * factor == W_hr, "HR size does not match LR size with given factor!"

    # model fit
    kernel_hr_reshape = kernel_hr.reshape(C, H_lr, factor, W_lr, factor)
    kernel_lr = kernel_hr_reshape.mean(axis=(2, 4))  # C, H_lr, W_lr
    y = lst_lr.flatten()
    X = kernel_lr.reshape(C,-1).T
    regression_model.fit(X, y)

    # downscale the residual term
    lst_lr_prd = regression_model.predict(X).reshape(H_lr, W_lr)
    residual_lr = lst_lr - lst_lr_prd
    residual_hr = cv2.resize(residual_lr, dsize=(H_hr, W_hr), interpolation=itp_methods[residual_add_way])

    # apply the fitted model to HR kernels
    X_hr = kernel_hr.reshape(C, -1).T
    lst_hr_prd = regression_model.predict(X_hr).reshape(H_hr, W_hr)

    # final prediction
    lst_hr_final = lst_hr_prd + residual_hr

    return lst_hr_final