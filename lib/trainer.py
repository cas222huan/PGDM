import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from .resshift import resshift_diffusion
from .utils import *
from .dataset import MyDataset, get_all_files
from .dataset_groklst import GrokLST_Dataset
from omegaconf import OmegaConf
import copy
import logging
from tqdm import tqdm
import requests

'''
set seed for each process
this is helpful when random operations are used in the dataset's __getitem__ method
avoid duplication of data enhancement between different processes
'''
def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


class BaseTrainer:
    def __init__(self, config_base_path, config_var_path, var_name):
        # load config
        config_base = OmegaConf.load(config_base_path)
        config_var_all = OmegaConf.load(config_var_path)
        self.config = config = OmegaConf.merge(
            copy.deepcopy(config_base),
            config_var_all[var_name])

        set_seed(config.seed)

        if config.log:
            log_dir = f"log/{var_name}.log"
            logging.basicConfig(filename=log_dir, level=logging.INFO, format='%(asctime)s - %(message)s')
            self.logger = logging.getLogger()
        else:
            self.logger = None

        self.ntfy = config.ntfy
        self.var_name = var_name
        self.n_epochs = config.train.n_epochs
        self.patience = config.train.patience
        self.device = config.device
        self.task_type = config.task_type # landsat or groklst

        if self.task_type == 'landsat_cn20':
            # build dataset and dataloaders
            indices_train = np.loadtxt('dataset/train.txt', dtype=int)
            indices_val = np.loadtxt('dataset/val.txt', dtype=int)
            file_path_train = get_all_files(config.folder_landsat_cn20, indices=indices_train)
            file_path_val = get_all_files(config.folder_landsat_cn20, indices=indices_val)
            self.trainset_landsat_cn20 = MyDataset(file_path_all=file_path_train, mode='train', **config.dataset)
            self.valset_landsat_cn20 = MyDataset(file_path_all=file_path_val, mode='test', **config.dataset)
            self.trainloader_landsat_cn20 = DataLoader(self.trainset_landsat_cn20, **config.dataloader_train, worker_init_fn=worker_init_fn)
            self.valloader_landsat_cn20 = DataLoader(self.valset_landsat_cn20, **config.dataloader_val, worker_init_fn=worker_init_fn)
        elif self.task_type == 'groklst':
            self.load_groklst_dataset(zoom=8)

        # build model, optimizer, scheduler
        model_type = config.model_type
        self.model = build_model(model_type, **config.model_params[model_type]).to(self.device)
        self.optimizer = build_optimizer(config.train.optimizer.type, self.model, **config.train.optimizer.params)
        self.scheduler = build_scheduler(config.train.scheduler.type, self.optimizer, **config.train.scheduler.params)
        self.loss_fn = get_loss_fn(config.loss_type)
        self.ckpt_save_path = f'models/{var_name}_best.pth'
    
    def _update_scheduler(self, loss_val):
        if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(loss_val)
        else:
            self.scheduler.step()

    def _log_start(self):
        if self.logger is not None:
            self.logger.info("=====================Start Training=====================")
            self.logger.info(f"config: {self.var_name}")
            self.logger.info(f"description: {self.config.description}")
            self.logger.info("Epoch, Loss_train, Loss_val (K)")

    def _log_epoch(self, epoch, loss_train, loss_val):
        log_str = f"{epoch}, {loss_train:.8f}, {loss_val:.4f}"
        if self.logger is not None:
            self.logger.info(log_str)

    def _send_notification(self, message):
        if self.logger is not None:
            self.logger.info(message)
        if self.ntfy is not None:
            requests.post(self.ntfy, data=message.encode(encoding='utf-8'))

    def _check_early_stopping(self, epoch, loss_val, count_not_decrease, loss_val_min):
        if loss_val < loss_val_min:
            count_not_decrease = 0
            loss_val_min = loss_val
            torch.save(self.model.state_dict(), self.ckpt_save_path)
        else:
            count_not_decrease += 1
        
        # early stop if needed
        if count_not_decrease >= self.patience:
            self._send_notification(f'Training finished with early stopping, best loss_val = {loss_val_min:.4f} K')
            return True, count_not_decrease, loss_val_min
        
        if epoch == self.n_epochs:
            self._send_notification(f'Training finished without early stopping, best loss_val = {loss_val_min:.4f} K')
        
        return False, count_not_decrease, loss_val_min
    
    def _save_checkpoint(self, epoch):
        # save model if needed
        if epoch % self.config.train.freq_save == 0:
            save_path_epoch = f"models/{self.var_name}_{epoch}.pth"
            torch.save(self.model.state_dict(), save_path_epoch)

    def denormalize_lst(self, lst_input):
        if self.task_type == 'landsat_cn20':
            lst_output = self.trainset_landsat_cn20.denormalize(lst_input, 'lst')
        elif self.task_type == 'groklst':
            lst_output = self.trainset_groklst.denormalize_lst(lst_input)
        return lst_output

    def reload_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path))
    
    def unfreeze_all(self):
        for param in self.model.parameters():
            param.requires_grad = True

    def load_testset_Landsat_CN20(self, fixed_drop_channels=None):
        indices_test = np.loadtxt('dataset/test.txt', dtype=int)
        file_path_test = get_all_files(self.config.folder_landsat_cn20, indices=indices_test)
        self.testset_landsat_cn20 = MyDataset(file_path_all=file_path_test, mode='test', fixed_drop_channels=fixed_drop_channels, **self.config.dataset)
        self.testloader_landsat_cn20 = DataLoader(self.testset_landsat_cn20, **self.config.dataloader_val, worker_init_fn=worker_init_fn)

    def load_testset_Landsat_GLB(self, fixed_drop_channels=None, indices=None):
        file_path = get_all_files(self.config.folder_landsat_glb, indices=indices)
        self.testset_landsat_glb = MyDataset(file_path_all=file_path, mode='test', fixed_drop_channels=fixed_drop_channels, **self.config.dataset)
        self.testloader_landsat_glb = DataLoader(self.testset_landsat_glb, **self.config.dataloader_val, worker_init_fn=worker_init_fn)

    def load_testset_ASTER_GLB(self, fixed_drop_channels=None, indices=None):
        file_path = get_all_files(self.config.folder_aster_glb, indices=indices)
        self.testset_aster_glb = MyDataset(file_path_all=file_path, mode='test', fixed_drop_channels=fixed_drop_channels, **self.config.dataset)
        self.testloader_aster_glb = DataLoader(self.testset_aster_glb, **self.config.dataloader_val, worker_init_fn=worker_init_fn)

    def load_dataset_tahoe(self, dataset_path):
        file_path = get_all_files(dataset_path)
        self.testset_tahoe = MyDataset(file_path_all=file_path, mode='test', fixed_drop_channels=[10], **self.config.dataset)
        self.testloader_tahoe = DataLoader(self.testset_tahoe, **self.config.dataloader_val, worker_init_fn=worker_init_fn)

    # def load_dataset_ASTER_GLB(self, indices_train, indices_val, indices_test):
    #     file_path_train = get_all_files(self.config.folder_aster_glb, indices=indices_train)
    #     file_path_val = get_all_files(self.config.folder_aster_glb, indices=indices_val)
    #     file_path_test = get_all_files(self.config.folder_aster_glb, indices=indices_test)
    #     self.trainset_aster_glb = MyDataset(file_path_all=file_path_train, mode='train', **self.config.dataset)
    #     self.valset_aster_glb = MyDataset(file_path_all=file_path_val, mode='test', **self.config.dataset)
    #     self.testset_aster_glb = MyDataset(file_path_all=file_path_test, mode='test', **self.config.dataset)

    #     self.trainloader_aster_glb = DataLoader(self.trainset_aster_glb, **self.config.dataloader_train, worker_init_fn=worker_init_fn)
    #     self.valloader_aster_glb = DataLoader(self.valset_aster_glb, **self.config.dataloader_val, worker_init_fn=worker_init_fn)
    #     self.testloader_aster_glb = DataLoader(self.testset_aster_glb, **self.config.dataloader_val, worker_init_fn=worker_init_fn)

    # def partial_freeze_model(self, act_modules):
    #     for param in self.model.parameters():
    #         param.requires_grad = False
        
    #     for module_name in act_modules:
    #         module = getattr(self.model, module_name, None)
    #         if module is not None:
    #             for param in module.parameters():
    #                 param.requires_grad = True
    #         else:
    #             print(f"Module {module_name} not found in the model.")

    def load_groklst_dataset(self, zoom=8):
        self.trainset_groklst = GrokLST_Dataset(root_dir=self.config.folder_groklst, txt_file='split/train.txt', zoom=zoom)
        self.valset_groklst = GrokLST_Dataset(root_dir=self.config.folder_groklst, txt_file='split/val.txt', zoom=zoom)
        self.testset_groklst = GrokLST_Dataset(root_dir=self.config.folder_groklst, txt_file='split/test.txt', zoom=zoom)

        self.trainloader_groklst = DataLoader(self.trainset_groklst, **self.config.dataloader_train, worker_init_fn=worker_init_fn)
        self.valloader_groklst = DataLoader(self.valset_groklst, **self.config.dataloader_val, worker_init_fn=worker_init_fn)
        self.testloader_groklst = DataLoader(self.testset_groklst, **self.config.dataloader_val, worker_init_fn=worker_init_fn)


class ResShiftTrainer(BaseTrainer):
    def __init__(self, config_base_path, config_var_path, var_name):
        super().__init__(config_base_path, config_var_path, var_name)
        self.n_timesteps = self.config.diffusion.n_timesteps
        # build diffusion model
        self.diffusion_model = self.build_diffusion_model(**self.config.diffusion)
    
    @staticmethod
    def build_diffusion_model(**kwargs):
        return resshift_diffusion(**kwargs)
    
    def set_seed_during_diffusion(self, seed):
        set_seed(seed)

    def train_step(self, epoch):
        print(f'Start training epoch {epoch}')
        loss_train_total = 0
        self.model.train()
        for lst_hr, _, lst_lr_itp, ref, ndxi, dem, lulc, mask in tqdm(self.trainloader):
            current_batch_size = lst_hr.shape[0]
            lst_hr, lst_lr_itp, ref, ndxi, dem, lulc, mask = map(lambda x: x.to(self.device), 
                                                                         [lst_hr, lst_lr_itp, ref, ndxi, dem, lulc, mask])

            t = torch.randint(0, self.n_timesteps, (current_batch_size, )).to(self.device)
            noise = torch.randn_like(lst_hr).to(self.device) # the same shape as targets
            x_t = self.diffusion_model.forward(lst_hr, lst_lr_itp, t, noise)
            x_start_prd = self.model(lst_lr_itp, ref, ndxi, dem, lulc, mask,
                                    self.diffusion_model.scale_input(inputs=x_t, t=t), 
                                    t)
            loss = self.loss_fn(x_start_prd, lst_hr)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_train_total += loss.item() * current_batch_size
        
        loss_train_mean = loss_train_total / len(self.trainloader.dataset)
        return loss_train_mean

    @torch.no_grad()
    def val_step(self, epoch):
        # evaluate the model performance on the validation set during model training
        print(f'Start validating at epoch {epoch}')
        self.model.eval()
        loss_val_total = 0
        for lst_hr, _, lst_lr_itp, ref, ndxi, dem, lulc, mask in tqdm(self.valloader):
            lst_hr, lst_lr_itp, ref, ndxi, dem, lulc, mask = map(lambda x: x.to(self.device), 
                                                                         [lst_hr, lst_lr_itp, ref, ndxi, dem, lulc, mask])

            lst_hr_prd, _ = self.diffusion_model.sample(model=self.model, 
                                                    n_timesteps = self.n_timesteps, 
                                                    lst_lr_itp = lst_lr_itp,
                                                    return_all_x_t=False,
                                                    ref=ref,
                                                    ndxi=ndxi,
                                                    dem=dem,
                                                    lulc=lulc,
                                                    mask=mask)

            # calculate actual accuracy metrics
            lst_hr_denorm = self.denormalize_lst(lst_hr)
            lst_hr_prd_denorm = self.denormalize_lst(lst_hr_prd)
            loss = self.loss_fn(lst_hr_denorm, lst_hr_prd_denorm)
            loss_val_total += loss.item() * lst_hr.shape[0]

        loss_val_mean = loss_val_total / len(self.valloader.dataset)
        return loss_val_mean

    def train(self, trainloader, valloader):
        self._log_start()
        count_not_decrease, loss_val_min = 0, np.inf
        # register dataloaders
        self.trainloader, self.valloader = trainloader, valloader
        for epoch in range(1, self.n_epochs+1):
            loss_train_current = self.train_step(epoch)
            loss_val_current = self.val_step(epoch)
            self._update_scheduler(loss_val_current)
            self._log_epoch(epoch, loss_train_current, loss_val_current)

            should_stop, count_not_decrease, loss_val_min = self._check_early_stopping(
                epoch, loss_val_current, count_not_decrease, loss_val_min)
            if should_stop:
                break

            self._save_checkpoint(epoch)

    @torch.no_grad()
    def evaluate_after_train(self, dataloader, return_lulc=False):
        # evaluate the model performance on the test set after model training
        self.model.eval()
        lst_hr_true_all, lst_hr_prd_all = [], []
        if return_lulc:
            lulc_all = []
        for lst_hr, _, lst_lr_itp, ref, ndxi, dem, lulc, mask in tqdm(dataloader):
            lst_hr, lst_lr_itp, ref, ndxi, dem, lulc, mask = map(lambda x: x.to(self.device), 
                                                                         [lst_hr, lst_lr_itp, ref, ndxi, dem, lulc, mask])

            lst_hr_prd, _ = self.diffusion_model.sample(model=self.model, 
                                                    n_timesteps=self.n_timesteps, 
                                                    lst_lr_itp = lst_lr_itp,
                                                    return_all_x_t=False,
                                                    ref=ref,
                                                    ndxi=ndxi,
                                                    dem=dem,
                                                    lulc=lulc,
                                                    mask=mask)
            
            lst_hr_true_all.append(lst_hr.cpu().numpy().squeeze(1))
            lst_hr_prd_all.append(lst_hr_prd.cpu().numpy().squeeze(1))
            if return_lulc:
                lulc_all.append(lulc.cpu().numpy().squeeze(1))

        lst_hr_prd_all = np.concatenate(lst_hr_prd_all, axis=0)
        lst_hr_true_all = np.concatenate(lst_hr_true_all, axis=0)
        lst_hr_prd_all = self.denormalize_lst(lst_hr_prd_all)
        lst_hr_true_all = self.denormalize_lst(lst_hr_true_all)

        if return_lulc:
            lulc_all = np.concatenate(lulc_all, axis=0)
            return lst_hr_true_all, lst_hr_prd_all, lulc_all
        else:
            return lst_hr_true_all, lst_hr_prd_all
    


class MocolskTrainer(BaseTrainer):
    def __init__(self, config_base_path, config_var_path, var_name):
        super().__init__(config_base_path, config_var_path, var_name)
    
    def train_step(self, epoch):
        print(f'Start training epoch {epoch}')
        loss_train_total = 0
        self.model.train()
        for lst_hr, lst_lr, _, ref, ndxi, dem, _, _ in tqdm(self.trainloader):
            gui = torch.cat((dem, ref, ndxi), axis=1)
            lst_hr, lst_lr, gui = map(lambda x: x.to(self.device), [lst_hr, lst_lr, gui])
            lst_hr_prd = self.model(lst_lr, gui)
            loss = self.loss_fn(lst_hr_prd, lst_hr)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_train_total += loss.item() * lst_hr.shape[0]

        loss_train_mean = loss_train_total / len(self.trainloader.dataset)
        return loss_train_mean

    @torch.no_grad()
    def val_step(self, epoch):
        # evaluate the model performance on the validation set during model training
        print(f'Start validating at epoch {epoch}')
        self.model.eval()
        loss_val_total = 0
        for lst_hr, lst_lr, _, ref, ndxi, dem, _, _ in tqdm(self.valloader):
            gui = torch.cat((dem, ref, ndxi), axis=1)
            lst_hr, lst_lr, gui = map(lambda x: x.to(self.device), [lst_hr, lst_lr, gui])
            lst_hr_prd = self.model(lst_lr, gui)

            # calculate actual accuracy metrics
            lst_hr_denorm = self.denormalize_lst(lst_hr)
            lst_hr_prd_denorm = self.denormalize_lst(lst_hr_prd)
            loss = self.loss_fn(lst_hr_denorm, lst_hr_prd_denorm)
            loss_val_total += loss.item() * lst_hr.shape[0]

        loss_val_mean = loss_val_total / len(self.valloader.dataset)
        return loss_val_mean

    def train(self, trainloader, valloader):
        self._log_start()
        count_not_decrease, loss_val_min = 0, np.inf
        # register dataloaders
        self.trainloader, self.valloader = trainloader, valloader
        for epoch in range(1, self.n_epochs+1):
            loss_train_current = self.train_step(epoch)
            loss_val_current = self.val_step(epoch)
            self._update_scheduler(loss_val_current)
            self._log_epoch(epoch, loss_train_current, loss_val_current)

            should_stop, count_not_decrease, loss_val_min = self._check_early_stopping(
                epoch, loss_val_current, count_not_decrease, loss_val_min)
            if should_stop:
                break

            self._save_checkpoint(epoch)

    @torch.no_grad()
    def evaluate_after_train(self, dataloader):
        self.model.eval()
        lst_hr_true_all, lst_hr_prd_all = [], []
        for lst_hr, lst_lr, _, ref, ndxi, dem, _, _ in tqdm(dataloader):
            gui = torch.cat((dem, ref, ndxi), axis=1)
            lst_hr, lst_lr, gui = map(lambda x: x.to(self.device), [lst_hr, lst_lr, gui])
            lst_hr_prd = self.model(lst_lr, gui)

            lst_hr_true_all.append(lst_hr.cpu().numpy().squeeze())
            lst_hr_prd_all.append(lst_hr_prd.cpu().numpy().squeeze())

        lst_hr_prd_all = np.concatenate(lst_hr_prd_all, axis=0)
        lst_hr_true_all = np.concatenate(lst_hr_true_all, axis=0)
        lst_hr_prd_all = self.denormalize_lst(lst_hr_prd_all)
        lst_hr_true_all = self.denormalize_lst(lst_hr_true_all)
        return lst_hr_true_all, lst_hr_prd_all