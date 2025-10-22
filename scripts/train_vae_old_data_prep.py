'''
File: train_vae.py
Author: Guido Di Federico (code is based on the implementation available at https://github.com/Project-MONAI/tutorials/tree/main/generative and https://github.com/huggingface/diffusers/)
Description: Script to train a variational autoencoder (VAE) to learn the mapping between geomodel space and low-dimensional latent space for latent diffusion models
Note: requires Python package "monai" or "monai-generative" to load VAE model and dataloaders
'''


# Import packages

# General imports
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

# Monai and diffusers modules
import monai
from monai import transforms
from monai.data import DataLoader, Dataset
from monai.utils import first, set_determinism
from generative.networks.nets import AutoencoderKL, DiffusionModelUNet

# Set directories
imgs_dir          =  '../data/imgs/'
trained_vae_dir = '../trained_vae/'

if not os.path.exists(trained_vae_dir):
    os.makedirs(trained_vae_dir)
    
# Choose device
#device = torch.device("cpu")
device = torch.device("cuda")



# Load dataset
geomodels_dataset = [{"image": imgs_dir + img} for  img in os.listdir(imgs_dir)][:4000]
N_data            = len(geomodels_dataset)

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

# Set hard data conditioning points (first two coordinates are (x,y) points and third coordinate the pixel value)
hard_data_locations = np.array([[7,7], [7,31], [7,55], [55,7], [55,31], [55,55]])


# Initiate variational autoendocder (VAE) model
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

# Train the VAE on three loss terms: (1) reconstruction loss, (2) K-L divergence loss, (3) hard data facies loss

# Training parameters
n_epochs      = 1000
val_interval  = 10
save_interval = 100
kl_weight     = 1e-6
hd_weight     = 1e-2


# Gradient parameters (optimizer and scaler)
optimizer = torch.optim.Adam(autoencoderkl.parameters(), lr=1e-4)
scaler = torch.cuda.amp.GradScaler()

# Training loop

epoch_losses = []
val_losses   = []

for epoch in range(n_epochs):
        
    autoencoderkl.train()
    epoch_recon_loss = 0
    epoch_kl_loss = 0
    epoch_hd_loss = 0
    epoch_loss = 0
    progress_bar = tqdm(enumerate(m_train_loader), total=len(m_train_loader), ncols=100)
    progress_bar.set_description(f"Epoch {epoch}")

    for step, batch in progress_bar:
        m_batch = batch["image"].to(device)
        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=True):
            
            reconstruction, z_mu, z_sigma = autoencoderkl(m_batch)
            recons_loss = F.l1_loss(reconstruction.float(), m_batch.float())

            reconstruction_hd = [reconstruction[...,loc[0],loc[1]] for loc in hard_data_locations]
            reconstruction_hd_vector =  torch.stack(reconstruction_hd, dim=0).flatten()
            m_batch_hd = [m_batch[...,loc[0],loc[1]] for loc in hard_data_locations]
            m_batch_hd_vector = torch.stack(m_batch_hd, dim=0).flatten()
            hd_loss =  F.mse_loss(m_batch_hd_vector, reconstruction_hd_vector)


            kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3])
            kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
            
            loss_tot = recons_loss + (kl_weight * kl_loss) + (hd_weight * hd_loss)


        scaler.scale(loss_tot).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_recon_loss += recons_loss.item()
        epoch_kl_loss += kl_loss.item() * kl_weight
        epoch_hd_loss += hd_loss.item() * hd_weight
        
        epoch_loss += loss_tot.item()

        progress_bar.set_postfix(
            {
                "recons_loss": epoch_recon_loss / (step + 1),
                "kl_loss": epoch_kl_loss / (step + 1),
                "hd_loss": epoch_hd_loss / (step + 1),
            }
        )
    
    epoch_losses.append(epoch_loss / (step + 1))
    
    if (epoch + 1) % save_interval == 0:
        torch.save(autoencoderkl.state_dict(), f'{trained_vae_dir}' + f'/vae_epoch_{epoch + 1}.pt')

    if (epoch + 1) % val_interval == 0:
        autoencoderkl.eval()
        val_loss = 0
        with torch.no_grad():
            for val_step, batch in enumerate(m_val_loader, start=1):
                m_batch = batch["image"].to(device)

                with autocast(enabled=True):
                    reconstruction, z_mu, z_sigma = autoencoderkl(m_batch)
                    recons_loss = F.l1_loss(reconstruction.float(), m_batch.float())

                    reconstruction_hd = [reconstruction[...,loc[0],loc[1]] for loc in hard_data_locations]
                    reconstruction_hd_vector =  torch.stack(reconstruction_hd, dim=0).flatten()
                    m_batch_hd = [m_batch[...,loc[0],loc[1]] for loc in hard_data_locations]
                    m_batch_hd_vector = torch.stack(m_batch_hd, dim=0).flatten()
                    hd_loss =  F.mse_loss(m_batch_hd_vector, reconstruction_hd_vector)


                    kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3])
                    kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

                    loss_g = recons_loss + (kl_weight * kl_loss) + (hd_weight * hd_loss)


                val_loss += loss_g.item()

        val_loss /= val_step
        val_losses.append(val_loss)
        print(f"epoch {epoch + 1} val loss: {val_loss:.4f}")
        
progress_bar.close()
