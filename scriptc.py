
import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
import torch.nn.functional as F
import math

import torch.nn as nn  # Import nn module

class SimpleUnet(nn.Module):
    def __init__(self):
        super(SimpleUnet, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # Add a final convolutional layer to adjust the number of channels
        self.conv_out = nn.Conv2d(128, 3, kernel_size=1)  # 1x1 convolution to change channels to 3

    def forward(self, x, t):

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
 
        x = self.conv_out(x) # Pass through the final convolutional layer
        return x # Replace with actual output


# Ensure the results directory exists
results_dir = "results"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Checkpoint functions
def save_checkpoint(model, optimizer, epoch, step, path="model_checkpoint.pth"):
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path} at epoch {epoch}, step {step}")

def load_checkpoint(path, model, optimizer):
    if os.path.isfile(path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        step = checkpoint.get('step', 0)  # Default step to 0 if not present
        print(f"Checkpoint loaded from {path} (epoch {epoch}, step {step})")
        return epoch, step
    else:
        print(f"No checkpoint found at {path}")
        return 0, 0  # Start from epoch 0, step 0 if no checkpoint exists

# Device configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Dataset and dataloader parameters
dataroot = r'C:\Users\Anshul Chauhan\Desktop\af\fake_faces' # Path to your dataset folder , which would be in fake_faces folder
image_size = 128
batch_size = 16

# Initialize dataset and dataloader
dataset = datasets.ImageFolder(
    root=dataroot,
    transform=transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
)

# Check if the dataset is empty
if len(dataset) == 0:
    raise ValueError(f"No valid images found in the dataset directory: {dataroot}")

dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0
)

# Linear beta schedule
def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    return torch.linspace(start, end, timesteps)

T = 300
beta = linear_beta_schedule(timesteps=T).to(device)
alpha = 1 - beta
alpha_hat = torch.cumprod(alpha, axis=0)

alpha_hat_mean_term = torch.sqrt(alpha_hat)
alpha_hat_vari_term = torch.sqrt(1 - alpha_hat)

# Forward diffusion
def forward_diffusion(x_0, timestep):
    x_0 = x_0.to(device)
    epsilon = torch.randn_like(x_0, device=device)
    return alpha_hat_mean_term[timestep] * x_0 + alpha_hat_vari_term[timestep] * epsilon, epsilon

# Neural network definitions

# Initialize the model and optimizer
model = SimpleUnet().to(device)
optimizer = torch.optim.Adam(model.parameters())

# Load checkpoint if it exists
checkpoint_path = "model_checkpoint.pth"
start_epoch, start_step = 0, 0
if os.path.exists(checkpoint_path):
    start_epoch, start_step = load_checkpoint(checkpoint_path, model, optimizer)

# Training loop
epochs = 50000
global_step = start_step  # Track global step across epochs

for epoch in range(start_epoch, epochs):
    print(f'Epoch {epoch}')
    for step, batch in enumerate(dataloader):
        optimizer.zero_grad()
        t = torch.randint(0, T, (1,), device=device).long()

        # Forward diffusion to generate noisy input
        x_noisy, _ = forward_diffusion(batch[0], t)

        # Model prediction
        model_output = model(x_noisy, t)

        # Compute loss
        target = batch[0].to(device)
        loss = F.l1_loss(model_output, target)
        print(f"Epoch {epoch}, Step {global_step} | Loss: {loss.item()}")

        # Backpropagation
        loss.backward()
        optimizer.step()

        # Save generated images periodically
        if global_step % 10 == 0:
            save_path = os.path.join(results_dir, f"epoch_{epoch}_step_{global_step}.png")
            save_image(model_output, save_path, normalize=True)
            print(f"Saved generated images to {save_path}")

            # Save checkpoint
            save_checkpoint(model, optimizer, epoch, global_step, checkpoint_path)

        global_step += 1  # Increment global step
