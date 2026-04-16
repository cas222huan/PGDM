import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from .moco.mocolsk_net import MoCoLSKNet
from .models import denosing_unet
# from .models import geo_unet, flexnet
# from .swinfusion import SwinFusion

# set seeds for reproducibility
def set_seed(seed:int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # for multi-GPU training

def build_model(model_type, **kwargs):
    model_registry = {'unet': denosing_unet,
                      'mocolsk': MoCoLSKNet}

    if model_type not in model_registry:
        raise ValueError(f"Model type '{model_type}' is not recognized. Available models: {list(model_registry.keys())}")

    return model_registry[model_type](**kwargs)

def build_optimizer(optimizer_type, model, **kwargs):
    optimizer_registry = {
        'adam': optim.Adam,
        'adamw': optim.AdamW,
        'sgd': optim.SGD
    }

    if optimizer_type not in optimizer_registry:
        raise ValueError(f"Optimizer type '{optimizer_type}' is not recognized. Available optimizers: {list(optimizer_registry.keys())}")

    return optimizer_registry[optimizer_type](model.parameters(), **kwargs)

def build_scheduler(scheduler_type, optimizer, **kwargs):
    scheduler_registry = {
        'reduce': optim.lr_scheduler.ReduceLROnPlateau,
        'cosine': optim.lr_scheduler.CosineAnnealingLR,
    }

    if scheduler_type not in scheduler_registry:
        raise ValueError(f"Scheduler type '{scheduler_type}' is not recognized. Available schedulers: {list(scheduler_registry.keys())}")

    return scheduler_registry[scheduler_type](optimizer, **kwargs)

def get_loss_fn(loss_type, **kwargs):
    loss_registry = {
        'mse': nn.MSELoss(),
        'mae': nn.L1Loss()
    }

    if loss_type not in loss_registry:
        raise ValueError(f"Loss type '{loss_type}' is not recognized. Available losses: {list(loss_registry.keys())}")

    return loss_registry[loss_type]

def calculate_spectral_indices(ref_b1, ref_b2):
    # NDVI: NIR and R
    # NDWI: G and NIR
    # NDBI: SWIR1 and NIR (similar to # NDMI: NIR and SWIR1)
    return (ref_b1 - ref_b2) / (ref_b1 + ref_b2 + 1e-10)  # Adding a small constant to avoid division by zero