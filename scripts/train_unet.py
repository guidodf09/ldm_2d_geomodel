'''
File: train_unet.py
Author: Guido Di Federico (code is based on the implementation available at https://github.com/Project-MONAI/tutorials/tree/main/generative and https://github.com/huggingface/diffusers/)
Description: Script to train a U-net to learn the de-noising process in the latent space of latent diffusion models
Note: differently from train_unet_old_data_prep.py, it does loads data automatically as a numpy array available at https://drive.google.com/drive/folders/1JCaaaJOvfReaqPbIBVtVnPAH7TVc4AA5?usp=sharing
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
data_path        = '../data/m_petrel.npy'
trained_unet_dir = '../trained_unet/'

if not os.path.exists(trained_unet_dir):
    os.makedirs(trained_unet_dir)
    
# Choose device
#device = torch.device("cpu")
device = torch.device("cuda")

# Load dataset
geomodel_dataset  = np.load(data_path).astype(np.float32)[:4000]
N_data            = geomodel_dataset.shape[0]

# Split dataset
train_split       = 0.7
val_split         = 0.2
test_split        = 1 - train_split - val_split
batch_size        = 16


# Wrap into MONAI-style dicts 
train_list = [{"image": train_models[i]} for i in range(train_models.shape[0])]
val_list   = [{"image": val_models[i]} for i in range(val_models.shape[0])]
test_list  = [{"image": test_models[i]} for i in range(test_models.shape[0])]

# Define transforms
default_transforms = transforms.Compose([
    transforms.ToTensord(keys=["image"]),
])

# Create datasets
m_train_ds = Dataset(data=train_list, transform=default_transforms)
m_val_ds   = Dataset(data=val_list,   transform=default_transforms)
m_test_ds  = Dataset(data=test_list,  transform=default_transforms)

# Create dataloaders
m_train_loader = DataLoader(m_train_ds, batch_size=batch_size, shuffle=True)
m_val_loader   = DataLoader(m_val_ds,   batch_size=batch_size, shuffle=False)
m_test_loader  = DataLoader(m_test_ds,  batch_size=batch_size, shuffle=False)


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
scheduler = DDPMScheduler(num_train_timesteps=1000, schedule="scaled_linear_beta", beta_start=0.0015, beta_end=0.0195, clip_sample=True)
#scheduler = DDIMScheduler(num_train_timesteps=1000, schedule="scaled_linear_beta", beta_start=0.0015, beta_end=0.0195, clip_sample=True) #Use this for inference, with  scheduler.set_timesteps(num_inference_steps=XXX)

# Compute scaling factor for non-perfectly Gaussian VAE latent spaces
example_data = first(m_train_loader)

with torch.no_grad():
    with autocast(enabled=True):
        z = autoencoderkl.encode_stage_2_inputs(example_data["image"].to(device))

scale_factor = 1 / torch.std(z)


inferer = LatentDiffusionInferer(scheduler, scale_factor=scale_factor)
optimizer = torch.optim.Adam(unet.parameters(), lr=1e-4)



# Training parameters
n_epochs      = 1000
val_interval  = 20
save_interval = 100

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
    
    if (epoch + 1) % save_interval == 0:
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
