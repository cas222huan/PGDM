import torch
import numpy as np
import math
from einops import rearrange

# only used when getitem include randomness
def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def _extract_into_tensor(arr:np.ndarray, timesteps:torch.Tensor, shape_str:str='b -> b 1 1 1') -> torch.Tensor:
    """
    Extract values from a 1-D numpy array for a batch of indices.
    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :return: a tensor of shape (e.g., [batch_size, 1, 1, 1])
    """
    result = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float() # float32
    return rearrange(result, shape_str)

def get_eta_schedule(
        n_timesteps,
        min_noise_level,
        etas_end=0.99,
        kappa=2.0,
        power=0.3):
    # note: etas_start and etas_end are the square root values
    etas_start = min(min_noise_level / kappa, min_noise_level)
    increaser = math.exp(1/(n_timesteps-1)*math.log(etas_end/etas_start))
    base = np.ones([n_timesteps, ]) * increaser
    power_timestep = np.linspace(0, 1, n_timesteps, endpoint=True)**power
    power_timestep *= (n_timesteps-1)
    sqrt_etas = np.power(base, power_timestep) * etas_start
    return sqrt_etas


class resshift_diffusion():
    def __init__(self, *, 
                 n_timesteps:int=15, 
                 kappa:float=2.0, 
                 sqrt_etas=None,
                 etas_end:float=0.99, 
                 min_noise_level:float=0.04, 
                 power:float=0.3, 
                 normalize_input:bool=True):

        self.sqrt_etas = np.array(sqrt_etas) if sqrt_etas is not None else get_eta_schedule(n_timesteps, min_noise_level, etas_end, kappa, power)
        self.etas = self.sqrt_etas**2
        self.kappa = kappa
        self.normalize_input = normalize_input

        self.n_timesteps = n_timesteps
        self.etas_prev = np.append(0.0, self.etas[:-1])
        self.alpha = self.etas - self.etas_prev # alpha_t = eta_t - eta_{t-1} for t > 1, else alpha_1 = eta_1

        self.posterior_mean_coef1 = self.etas_prev / self.etas
        self.posterior_mean_coef2 = self.alpha / self.etas
        self.posterior_variance = self.kappa**2 * self.etas_prev / self.etas * self.alpha
        self.posterior_variance_clipped = np.append(
                self.posterior_variance[1], self.posterior_variance[1:])
        # log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        # log-exp can provide better numerical stability especially for small values
        ## maybe we can directly calculate the sqrt values?
        self.posterior_log_variance_clipped = np.log(self.posterior_variance_clipped)

    def forward(self, x_start:torch.Tensor, y:torch.Tensor, t:torch.Tensor, noise=None):
        # Forward diffusion step: q (x_t | x_0, y)
        if noise is None:
            noise = torch.randn_like(x_start)
        etas_t = _extract_into_tensor(self.etas, t)
        sqrt_etas_t = _extract_into_tensor(self.sqrt_etas, t)
        x_t = x_start + etas_t * (y - x_start) + self.kappa * sqrt_etas_t * noise
        return x_t
    
    def sample_step(self, x_start:torch.Tensor, x_t:torch.Tensor, t:torch.Tensor, noise=None):
        # Reverse diffusion step: p (x_{t-1} | x_t, x_0, y)
        # x_0(x_start) is predicted from the neural network
        if noise is None:
            noise = torch.randn_like(x_start)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
        ) # no noise when t=0 (corresponds to t=1 in the original paper, as eta_0 is not defined)

        posterior_mean_coef1_t = _extract_into_tensor(self.posterior_mean_coef1, t)
        posterior_mean_coef2_t = _extract_into_tensor(self.posterior_mean_coef2, t)
        posterior_log_variance_clipped_t = _extract_into_tensor(self.posterior_log_variance_clipped, t)
        x_t_prev = (posterior_mean_coef1_t * x_t 
                    + posterior_mean_coef2_t * x_start 
                    + nonzero_mask * torch.exp(posterior_log_variance_clipped_t * 0.5) * noise)
        return x_t_prev
    
    def scale_input(self, inputs, t):
        if self.normalize_input:
            # 3-sigma
            inputs_max = _extract_into_tensor(self.sqrt_etas, t) * self.kappa * 3 + 1
            inputs_norm = inputs / inputs_max
        else:
            inputs_norm = inputs
        return inputs_norm
    
    def estimate_x_end(self, y, noise=None):
        if noise is None:
            noise = torch.randn_like(y)
        t = torch.tensor([self.n_timesteps - 1, ] * y.shape[0], device=y.device).long()
        return y + _extract_into_tensor(self.kappa * self.sqrt_etas, t) * noise

    @torch.no_grad()
    def sample(self, *, model, lst_lr_itp, n_timesteps, return_all_x_t:bool=False, **model_kwargs):
        batch_size = lst_lr_itp.shape[0]
        x_t = self.estimate_x_end(lst_lr_itp)
        x_t_all = [x_t.clone()] if return_all_x_t else None

        model.eval()
        timesteps = range(n_timesteps-1, -1, -1)
        for t in timesteps:
            t_tensor = torch.full((batch_size,), t, dtype=torch.long).to(lst_lr_itp.device)
            x_start_prd = model(lst_lr_itp=lst_lr_itp, **model_kwargs,
                                xt=self.scale_input(inputs=x_t, t=t_tensor), 
                                t=t_tensor)
            noise = torch.randn_like(x_start_prd)
            x_t = self.sample_step(x_start=x_start_prd, x_t=x_t, t=t_tensor, noise=noise)
            if return_all_x_t:
                x_t_all.append(x_t.clone())
        
        return (x_t, x_t_all) if return_all_x_t else (x_t, None)
