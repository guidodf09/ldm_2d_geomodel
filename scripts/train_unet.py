'''
File: train_unet.py
Author: Guido Di Federico (code is based on the implementation available at https://github.com/Project-MONAI/tutorials/tree/main/generative and https://github.com/huggingface/diffusers/)
Description: Script to train a U-net to learn the de-noising process in the latent space of latent diffusion models
Note: requires Python package "monai" or "monai-generative" to load 2D U-net model and dataloaders
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

# Set directories
imgs_dir          =  '../data/imgs/'
trained_unet_dir = '../trained_unet/'

if not os.path.exists(trained_unet_dir):
    os.makedirs(trained_unet_dir)
    
# Choose device
#device = torch.device("cpu")
device = torch.device("cuda")



# Load dataset
geomodels_dataset = [{"image": imgs_dir + img} for  img in os.listdir(imgs_dir)][:4000]
N_data            = len(geomodels_dataset)
image_size        = 64


# Split dataset
train_split       = 0.7
val_split         = 0.2
test_split        = 1 - train_split - val_split
batch_size        = 16

m_train_list    = geomodels_dataset[:int(N_data*train_split)]
m_val_list      = geomodels_dataset[int(len(m_train_list)):int(N_data*(1-test_split))+1]
m_test_list     = geomodels_dataset[int(-N_data*test_split):]

# Transform dataset

# Training set
train_transforms = transforms.Compose(
    [
        transforms.LoadImaged(keys=["image"]),
        transforms.EnsureChannelFirstd(keys=["image"]),
        transforms.ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True)]
)

m_train_ds = Dataset(data=m_train_list, transform=train_transforms)
m_train_loader = DataLoader(m_train_ds, batch_size=batch_size, shuffle=True)

# Validation set
val_transforms = transforms.Compose(
    [
        transforms.LoadImaged(keys=["image"]),
        transforms.EnsureChannelFirstd(keys=["image"]),
        transforms.ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True),
    ]
)
m_val_ds = Dataset(data=m_val_list, transform=val_transforms)
m_val_loader = DataLoader(m_val_ds, batch_size=batch_size, shuffle=True)

# Testing set
test_transforms = transforms.Compose(
    [
        transforms.LoadImaged(keys=["image"]),
        transforms.EnsureChannelFirstd(keys=["image"]),
        transforms.ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True),
    ]
)

m_test_ds = Dataset(data=m_test_list, transform=val_transforms)
m_test_loader = DataLoader(m_test_ds, batch_size=batch_size, shuffle=True)



# Initiate variational autoendocder (VAE) model and load pre-trained weights
trained_vae_dir = '../trained_vae/'
trained_vae_weights = trained_vae_dir + '/vae_epoch_1.pt'

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

# Initiate U-net model
unet = DiffusionModelUNet(
    spatial_dims=2,
    in_channels=1,
    out_channels=1,
    num_res_blocks=1,
    num_channels=(64, 128, 256),
    attention_levels=(False, True, True),
    num_head_channels=(0, 64, 128),
)
unet.to(device)


# Set noise scheduler to use for forward (noising) process
#scheduler = DDPMScheduler(num_train_timesteps=1000, schedule="linear_beta", beta_start=0.0001, beta_end=0.02)
scheduler = DDIMScheduler(num_train_timesteps=100, schedule="scaled_linear_beta", beta_start=0.0015, beta_end=0.0195, clip_sample=True) #Set to False for inference

# Compute scaling factor for non-perfectly Gaussian VAE latent spaces
example_data = first(m_train_loader)

with torch.no_grad():
    with autocast(enabled=True):
        z = autoencoderkl.encode_stage_2_inputs(example_data["image"].to(device))

scale_factor = 1 / torch.std(z)


inferer = LatentDiffusionInferer(scheduler, scale_factor=scale_factor)
optimizer = torch.optim.Adam(unet.parameters(), lr=1e-4)



# Training parameters
n_epochs      = 10
val_interval  = 5
save_interval = 10

# Train the U-net on the noise predicting function

epoch_losses  = []
val_losses    = []
scaler        = GradScaler()

for epoch in range(n_epochs):
    unet.train()
    autoencoderkl.eval()
    epoch_loss = 0
    progress_bar = tqdm(enumerate(m_train_loader), total=len(m_train_loader), ncols=100)
    progress_bar.set_description(f"Epoch {epoch}")
    
    for step, batch in progress_bar:
        images = batch["image"].to(device)
        optimizer.zero_grad(set_to_none=True)
        
        with autocast(enabled=True):
            z_mu, z_sigma = autoencoderkl.encode(images)
            z = autoencoderkl.sampling(z_mu, z_sigma) 
            
            noise = torch.randn_like(z).to(device)
            
            timesteps = torch.randint(0, inferer.scheduler.num_train_timesteps, (z.shape[0],), device=z.device).long()
            noise_pred = inferer(
                inputs=images, diffusion_model=unet, noise=noise, timesteps=timesteps, autoencoder_model=autoencoderkl
            )
            
            loss = F.mse_loss(noise_pred.float(), noise.float())

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()

        progress_bar.set_postfix({"loss": epoch_loss / (step + 1)})
    epoch_losses.append(epoch_loss / (step + 1))
    
    if (epoch + 1) % 10 == 0:
        torch.save(unet.state_dict(), f'{trained_unet_dir}' + f'/unet_epoch_{epoch + 1}.pt')

    if (epoch + 1) % val_interval == 0:
        unet.eval()
        val_loss = 0
        with torch.no_grad():
            for val_step, batch in enumerate(m_val_loader, start=1):
                images = batch["image"].to(device)

                with autocast(enabled=True):
                    z_mu, z_sigma = autoencoderkl.encode(images)
                    z = autoencoderkl.sampling(z_mu, z_sigma)

                    noise = torch.randn_like(z).to(device)
                    timesteps = torch.randint(
                        0, inferer.scheduler.num_train_timesteps, (z.shape[0],), device=z.device
                    ).long()
                    noise_pred = inferer(
                        inputs=images,
                        diffusion_model=unet,
                        noise=noise,
                        timesteps=timesteps,
                        autoencoder_model=autoencoderkl,
                    )

                    loss = F.mse_loss(noise_pred.float(), noise.float())

                val_loss += loss.item()
        val_loss /= val_step
        val_losses.append(val_loss)
        print(f"Epoch {epoch} val loss: {val_loss:.4f}")
progress_bar.close()
