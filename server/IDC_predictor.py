from cProfile import label
import math
import numpy as np
import os
from pathlib import Path
import pandas as pd
import json
import csv

import torchvision.models as models


import torch
from torch import optim, nn
import pytorch_lightning as pl
from torchvision import transforms
import torchvision.utils as vutils
from torchvision.datasets import CelebA
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import SGD, Adam, Adagrad
import torch.nn.functional as F

from PIL import Image

from models import BaseVAE
from models.types_ import *
from utils import data_loader

class IDC_Dataset(Dataset):
    def __init__(self, root, transform=None, target_transform=None, train=True):
        df = pd.read_csv(os.path.join(root, 'label.csv'))
        if train:
            self.img_labels = df.iloc[:202154, :]
        else:
            self.img_labels = df.iloc[202154:, :]
        self.img_dir = root
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, f'{self.img_labels.iloc[idx, 0]}')
        image = Image.open(img_path)
        # 
        label = self.img_labels.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
            
        return image, label


# extend the pytorch lightning module
class IDCPredictor(pl.LightningModule):

    def __init__(self) -> None:
        super(IDCPredictor, self).__init__()        
       
        self.model = models.resnet34(pretrained=True)
        # Replace last layer
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, 1)
        # model.compile(loss = 'binary_crossentropy', optimizer ='adam', metrics= ['accuracy'])
        
        self.params = {
                     "dataset": "IDC_regular_ps50_idx5",
                     "data_path": "./data",
                     'batch_size': 64,
                     'LR': 0.002
                 }

        if torch.cuda.is_available():
            device = torch.cuda.current_device()
        else:
            device = torch.device("cpu")
        self.curr_device = device
       
    @property
    def logger_folder(self):
        return f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/"

    # def freeze(self):
    #     # To freeze the residual layers
    #     for param in self.network.parameters():
    #         param.require_grad = False
    #     for param in self.network.fc.parameters():
    #         param.require_grad = True
    
    # def unfreeze(self):
    #     # Unfreeze all layers
    #     for param in self.network.parameters():
    #         param.require_grad = True

    
    def forward(self, input: Tensor, **kwargs) -> Tensor:
        
        return self.model(input, **kwargs)

    def loss_function(self,
                      results,
                      labels,
                      **kwargs) -> dict:
        return F.mse_loss(F.sigmoid(results), labels)


    def training_step(self, batch, batch_idx, optimizer_idx = 0):
        real_img, labels = batch

        self.curr_device = real_img.device

        results = self.forward(real_img)
        train_loss = self.loss_function(results, labels,
                                              M_N = self.params['batch_size']/ self.num_train_imgs,
                                              optimizer_idx=optimizer_idx,
                                              batch_idx = batch_idx)

        self.logger.experiment.log({key: val.item() for key, val in train_loss.items()})

        return train_loss

    def training_end(self, outputs):
        
        return {'loss': outputs['loss']}

    def validation_step(self, batch, batch_idx, optimizer_idx = 0):
        real_img, labels = batch


        self.curr_device = real_img.device
        results = self.forward(real_img, labels = labels)
        val_loss = self.loss_function(results, labels,
                                            M_N = self.params['batch_size']/ self.num_val_imgs,
                                            optimizer_idx = optimizer_idx,
                                            batch_idx = batch_idx)

        return val_loss

    def validation_end(self, outputs):
        
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        tensorboard_logs = {'avg_val_loss': avg_loss}
       

        print('val_loss: ', avg_loss.item())

        return {'val_loss': avg_loss, 'log': tensorboard_logs}


    def test_step(self, batch, batch_idx, optimizer_idx = 0):
        
        real_img, labels = batch

        
        self.curr_device = real_img.device

        results = self.forward(real_img, labels = labels)

        loss = self.loss_function(results, labels,
                                            M_N = self.params['batch_size']/ self.num_val_imgs,
                                            optimizer_idx = optimizer_idx,
                                            batch_idx = batch_idx)
        return loss

    def test_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        print('test_loss', avg_loss)

        return {'test_loss': avg_loss}


    def configure_optimizers(self):
        return Adam(
            self.parameters(), 
            lr=self.params['LR'],
            betas = (0.9, 0.999)
        )
        

    @data_loader
    def train_dataloader(self):
        print('start train data loading')
        root = os.path.join(self.params['data_path'], f"{self.params['dataset']}.npz")
       
        data = np.load(root, encoding='bytes')

        tensor = torch.from_numpy(data['x'])
        labels = torch.from_numpy(data['y'])
        # TODO: hard code a categorical transfer function



        train_kwargs = {'data_tensor':tensor, 'labels': labels}
        dset = IDC_Dataset
        train_data = dset(**train_kwargs, transform = self.data_transforms(), train=True)
        self.num_train_imgs = len(train_data)
        train_loader = DataLoader(train_data,
                                batch_size=self.params['batch_size'],
                                shuffle=True,
                                drop_last=True)

        print('end train data loading')
        return train_loader


    @data_loader
    def val_dataloader(self):
        root = os.path.join(self.params['data_path'], f"{self.params['dataset']}.npz")
       
        data = np.load(root, encoding='bytes')

        tensor = torch.from_numpy(data['x'])
        labels = torch.from_numpy(data['gt'])


        val_kwargs = {'data_tensor':tensor, 'labels': labels}
        dset = IDC_Dataset
        val_data = dset(**val_kwargs, transform = self.data_transforms(), train=False)
        self.num_val_imgs = len(val_data)
        val_loader = DataLoader(val_data,
                                batch_size=self.params['batch_size'],
                                shuffle=True,
                                drop_last=True)

        return val_loader
        
    
    @data_loader
    def test_dataloader(self):
        root = os.path.join(self.params['data_path'], f"{self.params['dataset']}.npz")
       
        data = np.load(root, encoding='bytes')

        tensor = torch.from_numpy(data['x'])
        labels = torch.from_numpy(data['gt'])


        test_kwargs = {'data_tensor':tensor, 'labels': labels}
        dset = IDC_Dataset
        test_data = dset(**test_kwargs, transform = self.data_transforms(), train=False)
        self.num_test_imgs = len(test_data)
        test_loader = DataLoader(test_data,
                                batch_size=self.params['batch_size'],
                                shuffle=False,
                                drop_last=False)

        return test_loader

    def data_transforms(self):
        norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]) # normalize to use the pretrained weights
        transform = transforms.Compose([transforms.RandomApply([transforms.RandomHorizontalFlip(1), transforms.RandomVerticalFlip(1)], 0.5),
                                        transforms.ToTensor(),
                                        norm
                                        ])
        return transform
