from cProfile import label
import math
import numpy as np
import os
from pathlib import Path
import pandas as pd
import json
import csv


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


from models.types_ import *
from utils import data_loader


class CustomTensorDataset(Dataset):
    def __init__(self, data_tensor, labels = None):
        self.data_tensor = data_tensor
        self.labels = labels

    def __getitem__(self, index):
            
        if torch.is_tensor(self.labels):
            return self.data_tensor[index], self.labels[index]
        return self.data_tensor[index], 0

    def __len__(self):
        return self.data_tensor.size(0)


# extend the pytorch lightning module
class ConceptAdaptor(pl.LightningModule):

    def __init__(self,
                 cat_num: int,  
                 input_size: list,
                 params: dict = {
                     "dataset": "",
                     "data_path": "./data",
                     'batch_size': 64,
                     'LR': 0.002,
                     'dim': 0
                 }) -> None:

        '''
        @cat_num: number of categories. 1 indicates a regression model
        '''
        super(ConceptAdaptor, self).__init__()
        
        [c, h, w] = input_size
        self.cat_num = cat_num
        kernel_size = 4
        stride = 2
        self.params = params

        if cat_num == 1:
            self.concept_adaptor = nn.Sequential(
                nn.Conv2d(c, 2, 4, 2),
                nn.BatchNorm2d(2),
                nn.MaxPool2d( (h-kernel_size)/stride + 1, (w-kernel_size)/stride + 1),
                nn.Flatten(),
                nn.Sigmoid()
            )
        
        else:
            self.concept_adaptor = nn.Sequential(
                nn.Conv2d(c, cat_num, 4, 2),
                nn.BatchNorm2d(cat_num),
                nn.MaxPool2d( (math.floor((h-kernel_size)/stride + 1), math.floor((w-kernel_size)/stride + 1))),
                nn.Flatten()
            )

        
        self.params = params

        if torch.cuda.is_available():
            device = torch.cuda.current_device()
        else:
            device = torch.device("cpu")
        self.curr_device = device
       
    @property
    def logger_folder(self):
        return f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/"

    
    def forward(self, input: Tensor, **kwargs) -> Tensor:
        
        return self.concept_adaptor(input, **kwargs)

    def loss_function(self,
                      results,
                      labels,
                      **kwargs) -> dict:
        if self.cat_num == 1:
            m = results[:, 0]
            log_var = results[:, 1]
            std = torch.exp(0.5 * log_var)
            dist = torch.distributions.Normal(m, std)
            return -dist.log_prob(labels).sum()
        else:
            return F.nll_loss(F.log_softmax(results, dim=1), labels)


    def training_step(self, batch, batch_idx, optimizer_idx = 0):
        real_img, labels = batch

        self.curr_device = real_img.device

        results = self.forward(real_img)
        train_loss = self.loss_function(results, labels,
                                              M_N = self.params['batch_size']/ self.num_train_imgs,
                                              optimizer_idx=optimizer_idx,
                                              batch_idx = batch_idx)


        return train_loss

    def training_end(self, outputs):
        
        return {'loss': outputs}

    def validation_step(self, batch, batch_idx, optimizer_idx = 0):
        real_img, labels = batch


        self.curr_device = real_img.device
        results = self.forward(real_img)
        val_loss = self.loss_function(results, labels,
                                            M_N = self.params['batch_size']/ self.num_val_imgs,
                                            optimizer_idx = optimizer_idx,
                                            batch_idx = batch_idx)

        return val_loss

    def validation_end(self, outputs):
        
        avg_loss = torch.stack(outputs).mean()
        tensorboard_logs = {'avg_val_loss': avg_loss}
       

        print('val_loss: ', avg_loss.item())

        return {'val_loss': avg_loss, 'log': tensorboard_logs}


    def test_step(self, batch, batch_idx, optimizer_idx = 0):
        
        real_img, labels = batch

        
        self.curr_device = real_img.device

        results = self.forward(real_img)

        loss = self.loss_function(results, labels,
                                            M_N = self.params['batch_size']/ self.num_val_imgs,
                                            optimizer_idx = optimizer_idx,
                                            batch_idx = batch_idx)
        return loss

    def test_end(self, outputs):
        avg_loss = torch.stack(outputs).mean()
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
        # labels = torch.from_numpy(data['gt'][:, self.params['dim']])
        mapper = np.vectorize(self.params['y_mapper'])
        labels = data['y'][:, self.params['dim']]
        labels = torch.from_numpy(mapper( labels ))
        # TODO: hard code a categorical transfer function



        train_kwargs = {'data_tensor':tensor, 'labels': labels}
        dset = CustomTensorDataset
        train_data = dset(**train_kwargs)
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

        mapper = np.vectorize(self.params['gt_mapper'])
        labels = data['gt'][:, self.params['dim']]
        labels = torch.from_numpy( mapper(labels))


        val_kwargs = {'data_tensor':tensor, 'labels': labels}
        dset = CustomTensorDataset
        val_data = dset(**val_kwargs)
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
        mapper = np.vectorize(self.params['gt_mapper'])
        labels = data['gt'][:, self.params['dim']]
        labels = torch.from_numpy( mapper(labels))

        test_kwargs = {'data_tensor':tensor, 'labels': labels}
        dset = CustomTensorDataset
        test_data = dset(**test_kwargs)
        self.num_test_imgs = len(test_data)
        test_loader = DataLoader(test_data,
                                batch_size=self.params['batch_size'],
                                shuffle=False,
                                drop_last=False)

        return test_loader

