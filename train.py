import sys
sys.path.append('./DiffusionFastForward/')

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from PIL import Image


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import imageio
from skimage import io
import os

from src import *

mpl.rcParams['figure.figsize'] = (8, 8)



# Define the Dataset
class ImageFolderDataset(Dataset):
    def __init__(self, folder_path,target_size=128):
        self.folder_path,self.target_size = folder_path,target_size
        self.image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = T.ToTensor()
        #self.image_files = self.image_files[:100]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.folder_path, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")
        image = image.resize((self.target_size,self.target_size), Image.BILINEAR)
        image = self.transform(image)
        return image

class ValDataset(Dataset):
    def __init__(self):
      pass
    def __len__(self):
      return(1)
    def __getitem__(self,idx):
      return {}


# Create the dataset
folder_path = "data/test"
dataset_train = ImageFolderDataset(folder_path)
dataset_val = ValDataset()
print(f"Dataset Instanciated with {len(dataset_train)} images.")



# define custom PL funciton to save images
def custom_on_validation_epoch_end(self,nrow = 2):
    assert nrow in [1,2,3], "nrow must be in 1,2,3. Otherwise will take too long"
    
    import os,torch
    from torchvision.utils import make_grid, save_image
    
    # generate Images with DDPM
    batch_size = nrow**2
    generated_images_tensor = self.model(batch_size=batch_size,shape=(128,128),verbose=True)
    generated_images_tensor = generated_images_tensor.clamp_(0, 1) # clamp

    # Create a 3x3 grid of the generated images
    image_grid = make_grid(generated_images_tensor, nrow=nrow)

    # Ensure the 'images' folder exists
    os.makedirs("images", exist_ok=True)

    # Save the grid of images as 'epoch_n.png'
    current_epoch = self.current_epoch+1
    file_path = f"example_images/epoch_v2_{current_epoch:05}.png"
    save_image(image_grid, file_path)
    #print(f"Saved generated image grid for epoch {current_epoch} at {file_path}")


def custom_validation_step(self, batch, batch_idx):     
    pass

def create_grid(self,nrow=2):
    assert nrow in [1,2,3], "nrow must be in 1,2,3. Otherwise will take too long"
    from torchvision.utils import make_grid
    from torchvision.transforms import ToPILImage
    
    # generate Images with DDPM
    b = nrow**2
    generated_images_tensor = self.model(batch_size=b,shape=(128,128),verbose=True)

    # Create a 3x3 grid of the generated images
    image_grid = make_grid(generated_images_tensor, nrow=nrow)
    pil_image = ToPILImage()(image_grid)
    pil_image = pil_image.resize((nrow*128*3,nrow*128*3), Image.BILINEAR) # make image larger
    return(pil_image)


# Model Saving Callback
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer

# Define the callback
checkpoint_callback = ModelCheckpoint(
    dirpath="checkpoints",    # Directory to save the model checkpoints
    filename="model_v2_{epoch}", # Filename template (epoch number will be appended)
    save_top_k=-1,             # S
    every_n_epochs=250         # Save the model after every n epoch
)



# Instanciate Model
model=PixelDiffusion(train_dataset=dataset_train,
                     valid_dataset=dataset_val,
                     num_timesteps=1000,
                     lr=1e-4,
                     batch_size=32)

# Set image saving funciton
model.on_validation_epoch_end = custom_on_validation_epoch_end.__get__(model)
model.validation_step = custom_validation_step.__get__(model)

# Save create Grid Function
model.create_grid = create_grid.__get__(model)
model = model.to("cuda")

# prepare PL trainer
trainer = pl.Trainer(
    max_epochs=10000,
    check_val_every_n_epoch=50,
    callbacks=[EMA(0.9999),
               checkpoint_callback],
    gpus = [0]
)



trainer.fit(model)



inf = False
if inf:
    for i in range(10):
        img = model.create_grid(nrow=3)
        