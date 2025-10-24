'''
File: generate.py
Author: Guido Di Federico (code is based on the implementation available at https://github.com/Project-MONAI/tutorials/tree/main/generative and https://github.com/huggingface/diffusers/)
Description: Script to generate new samples with a trained VAE and U-net (i.e., latent diffusion model)
'''


# Import packages

# General imports
import os
import numpy as np
import shutil
import tempfile
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from PIL import Image 
import cv2
import matplotlib.pyplot as plt 


# Monai and diffusers modules
import monai
from monai import transforms
from monai.data import DataLoader, Dataset
from monai.utils import first, set_determinism
from generative.inferers import LatentDiffusionInferer
from generative.networks.nets import AutoencoderKL, DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler, DDIMScheduler

# Choose device
#device = torch.device("cpu")
device = torch.device("cuda")

# Set parameters
Nx_latent    = 8
Ny_latent    = 8
n_steps      = 100
scale_factor =  1.
thresh1 = 0.2
thresh2 = 0.8


# Initiate variational autoendocder (VAE) model and load pre-trained weights
trained_vae_dir = '../trained_vae/'
trained_vae_weights = trained_vae_dir + '/vae_epoch_1000.pt'

autoencoderkl = AutoencoderKL(
    spatial_dims= 2,
    in_channels= 1,
    out_channels= 1,
    num_channels=(64, 128, 256, 512),
    latent_channels= 1,
    num_res_blocks= 1,
    norm_num_groups= 16,
    attention_levels= (False, False, False, True)
    )
autoencoderkl = autoencoderkl.to(device)
checkpoint    = torch.load(trained_vae_weights)
autoencoderkl.load_state_dict(checkpoint)
autoencoderkl.eval()


# Initiate U-net model and load pre-trained weights
trained_unet_dir = '../trained_unte/'
trained_unet_weights = trained_vae_dir + '/vae_epoch_1000.pt'


unet = DiffusionModelUNet(
    spatial_dims=2,
    in_channels=1,
    out_channels=1,
    num_res_blocks=1,
    num_channels=(64, 128, 256),
    attention_levels=(False, True, True),
    num_head_channels=(0, 64, 128),
)

unet = unet.to(device)
checkpoint    = torch.load(trained_unet_weights)
unet.load_state_dict(checkpoint)
unet.eval()


# Set noise scheduler to use for forward (noising) process
scheduler = DDPMScheduler(num_train_timesteps=1000, schedule = "scaled_linear_beta", beta_start=0.0015, beta_end=0.0195, clip_sample=True)
scheduler = DDIMScheduler(num_train_timesteps=1000, schedule="scaled_linear_beta", beta_start=0.0015, beta_end=0.0195, clip_sample=False) #Set to False for inference
scheduler.set_timesteps(num_inference_steps=n_steps)

# Initialize diffusion model inferer
inferer = LatentDiffusionInferer(scheduler, scale_factor=scale_factor)


# Simple function to tri-categorize generated samples

def model2tricat(model, thresh1, thresh2):
    """
    Convert a continuous model array into three categorical zones 
    based on two threshold values.

    Parameters:
        model (ndarray): Input continuous model array.
        thresh1 (float): Lower threshold separating category 0 and 0.5.
        thresh2 (float): Upper threshold separating category 0.5 and 1.

    Returns:
        ndarray: Tricategorical array with values {0, 0.5, 1}.
    """
    model_copy = np.copy(model)
    model_copy[model_copy < thresh1] = 0.
    model_copy[(model_copy >= thresh1) & (model_copy <= thresh2)] = 0.5
    model_copy[model_copy > thresh2] = 1.
    return model_copy


# Generate new samples
n_samples = 100
noise = torch.randn(size = (n_samples,1,Nx_latent, Ny_latent)).to(device)
generated_samples = inferer.sample(input_noise=noise,
                                   autoencoder_model=autoencoderkl,
                                   diffusion_model=unet,
                                   scheduler=scheduler).detach().cpu().numpy()[:,0]

generated_samples_tricat = model2tricat(generated_samples, thresh1, thresh2)
np.save('generated_samples.npy', generated_samples_tricat)
