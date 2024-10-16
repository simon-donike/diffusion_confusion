import sys
sys.path.append('./DiffusionFastForward/')

import torch
torch.set_float32_matmul_precision('medium')
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


from kornia.utils import image_to_tensor
import kornia.augmentation as KA

class SimpleImageDataset(Dataset):
    """Dataset returning images in a folder."""

    def __init__(self,
                 root_dir,
                 transforms=None,
                 paired=True,
                 return_pair=False):
        self.root_dir = root_dir
        self.transforms = transforms
        self.paired=paired
        self.return_pair=return_pair
        self.size=256
        
        # set up transforms
        if self.transforms is not None:
            if self.paired:
                data_keys=2*['input']
            else:
                data_keys=['input']

            self.input_T=KA.container.AugmentationSequential(
                *self.transforms,
                data_keys=data_keys,
                same_on_batch=False
            )   
        
        # check files
        supported_formats=['webp','jpg']        
        self.files=[el for el in os.listdir(self.root_dir) if el.split('.')[-1] in supported_formats]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()            

        img_name = os.path.join(self.root_dir,
                                self.files[idx])
        image = image_to_tensor(io.imread(img_name))/255

        if self.paired:
            c,h,w=image.shape
            slice=int(w/2)
            image2=image[:,:,slice:]
            image=image[:,:,:slice]
            if self.transforms is not None:
                out = self.input_T(image,image2)
                image=out[0][0]
                image2=out[1][0]
        elif self.transforms is not None:
            image = self.input_T(image)[0]

        if self.return_pair:
            image = F.interpolate(image.unsqueeze(0),size=(self.size,self.size),mode='bilinear').squeeze(0)
            image2 = F.interpolate(image2.unsqueeze(0),size=(self.size,self.size),mode='bilinear').squeeze(0)
            return image2,image
        else:
            image = F.interpolate(image.unsqueeze(0),size=(self.size,self.size),mode='bilinear').squeeze(0)
            return image
        
        
# Get Datasets
train_ds = SimpleImageDataset('data/maps/train',return_pair=True)
test_ds  = SimpleImageDataset('data/maps/val'  , return_pair=True)



# define custom PL funciton to save images
def custom_validation_step(self,batch,batch_idx):
    if type(batch)==tuple or type(batch)==list:
        condition=batch[0]
        gt = batch[1]
    else:
        condition=batch
    with torch.no_grad():
        pred=model(condition,sampler=self.ddim_sampler,verbose=True)
    pred = F.pad(pred, (2, 2, 2, 2), mode='constant', value=0)
    condition = F.pad(condition, (2, 2, 2, 2), mode='constant', value=0)
    output = torch.cat((condition, pred), dim=3)
    if type(batch)==tuple or type(batch)==list:
        gt = F.pad(gt, (2, 2, 2, 2), mode='constant', value=0)
        output = torch.cat((output, gt), dim=3)
    B, C, W2, H = output.shape
    output = torch.cat((output[0], output[1]), dim=1)
    output = output.permute(1, 2, 0)
    # Save the image
    output = output.cpu().numpy()
    output = np.clip(output, 0, 1)
    output = (output * 255).astype(np.uint8)
    imageio.imwrite(f"example_images_conditional/epoch_v2_{(self.current_epoch+1):05}.png", output)
    
    
    
def define_custom_sampler(self):
    # Set Up DDIM Sampling
    STEPS=200 # ddim steps
    ddim_sampler=DDIM_Sampler(STEPS,model.model.num_timesteps)
    model.ddim_sampler = ddim_sampler
    print("DDIM Sampler Defined.")
    

# Model Saving Callback
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer

# Define the callback
checkpoint_callback = ModelCheckpoint(
    dirpath="checkpoints_conditional",    # Directory to save the model checkpoints
    filename="model_v2_{epoch}", # Filename template (epoch number will be appended)
    save_top_k=-1,             # S
    every_n_epochs=250         # Save the model after every n epoch
)


# Instanciate Model
model=PixelDiffusionConditional(train_dataset=train_ds,
                                valid_dataset=test_ds,
                                lr=1e-4,
                                batch_size=16)
model.validation_step = custom_validation_step.__get__(model)
model.define_custom_sampler = define_custom_sampler.__get__(model)
model.define_custom_sampler()



trainer = pl.Trainer(
    max_epochs=10000,
    strategy='ddp',
    check_val_every_n_epoch=50,
    limit_val_batches=1,
    callbacks=[EMA(0.9999),
               checkpoint_callback],
    gpus = [1]
)



trainer.fit(model)

