HOW TO RUN

Install dependent packages - pip install -r requirements.txt
1. Run the download_images.py on your pc, it will save the images in faces folder in desktop. (set dataset image count on it- default 50000).
2. Create a fake_face director on same directory as scritc.py . fake_faces will contain "faces" folder (in which whole dataset images are).
3. Now to run main scriptc.py , first set the dataroot path to fake_faces folder. then run it.(path e.g. '/content/fake_face')

   after running main file it would create a result folder and a model_checkpoint.pth .
   As there is no model checkpoint for starting, it will be created automatically on first running.



DATASET
Dataset are all fake face images downloaded from https://thispersondoesnotexist.com/ .

The code implements a denoising diffusion model using a simplified U-Net architecture. The goal is to train the model to reverse the process of adding noise to images (a process called diffusion), ultimately generating new, clean images.

Step-by-Step Process
Input Data (Images):
- The model uses a dataset of images.
- Images are preprocessed by resizing, cropping, and normalizing them.
Add Noise to Images (Diffusion Process):
- The model takes an image and progressively adds noise over time (using the forward_diffusion function).
- The result is a noisy version of the image for a given timestep t.
Learn to Remove Noise (Reverse Process):
- The model, SimpleUnet, is trained to predict the noise added at a specific timestep t.
- By removing this predicted noise, the model reconstructs a cleaner image from the noisy one.
Training Objective:
- Minimize the difference between the original image and the reconstructed image (via L1 loss).
- This teaches the model to reverse the diffusion process step by step.
SimpleUnet Architecture

The SimpleUnet architecture is a U-shaped neural network designed to process images. Here's how it works:

Input Layer:
- Takes in the noisy image (x) and the timestep (t).
Time Embedding:
- The timestep t is converted into a meaningful representation using sinusoidal embeddings (SinusoidalPositionEmbeddings).
- This helps the model understand when in the diffusion process the image is.
Downsampling (Contracting Path):
- A sequence of convolutional layers reduces the spatial dimensions of the image while increasing feature depth.
- Captures 'what' is in the image at different scales.
Bottleneck:
- The central part of the U-Net processes the smallest representation of the image.
- Encodes the most abstract features of the image.
Upsampling (Expanding Path):
- A sequence of upsampling layers reconstructs the image to its original size.
- Combines high-level features (from bottleneck) with lower-level features (from earlier layers) using skip connections.
Output Layer:
- Produces a prediction for the noise that was added to the image.
- This noise prediction is subtracted from the noisy image to generate a cleaner version.
Detailed Workflow
Forward Pass:
- The model receives:
    - A noisy image (x).
    - The current timestep (t).
- The image is passed through the U-Net.
Skip Connections:
- During downsampling, intermediate features are saved and later re-used during upsampling.
- Helps preserve fine details of the image.
Output:
- The model predicts the noise added at the current timestep.
Loss Function:
- The predicted noise is compared to the actual noise (L1 loss).
- This trains the model to accurately predict noise.



![epoch_1_step_0](https://github.com/user-attachments/assets/8cbab556-0117-49f2-8b78-7b2af105b75c)















