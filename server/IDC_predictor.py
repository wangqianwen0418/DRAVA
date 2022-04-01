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
from pytorch_lightning import Trainer
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
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 1)
        # self.freeze()
        
        self.params = {
                     "dataset": "IDC_regular_ps50_idx5",
                     "data_path": "./data",
                     'batch_size': 64,
                     'LR': 0.0002
                 }

        if torch.cuda.is_available():
            device = torch.cuda.current_device()
        else:
            device = torch.device("cpu")
        self.curr_device = device
       
    @property
    def logger_folder(self):
        return f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/"

    def freeze(self):
        # To freeze the residual layers
        for param in self.model.parameters():
            param.require_grad = False
        for param in self.model.fc.parameters():
            param.require_grad = True
    
    def unfreeze(self):
        # Unfreeze all layers
        for param in self.model.parameters():
            param.require_grad = True

    
    def forward(self, input: Tensor, **kwargs) -> Tensor:
        
        return self.model(input, **kwargs)

    def loss_function(self,
                      results,
                      labels,
                      **kwargs) -> Tensor:

        # return F.mse_loss(torch.flatten(F.sigmoid(results)), labels.float())
        out = torch.flatten(torch.sigmoid(results))
        return F.binary_cross_entropy(out, labels.float())  
    
    def acc_function(self, results, labels, batch_size):
        acc = ((torch.flatten(results) > 0.5) == (labels==1) ).sum().item()/batch_size
        return torch.tensor(acc)


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
        acc = self.acc_function(results, labels, self.params['batch_size'])

        val_loss = self.loss_function(results, labels,
                                            M_N = self.params['batch_size']/ self.num_val_imgs,
                                            optimizer_idx = optimizer_idx,
                                            batch_idx = batch_idx)

        return {'loss': val_loss, 'acc': acc}

    def validation_end(self, outputs):

        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        acc = torch.stack([x['acc'] for x in outputs]).mean()
        tensorboard_logs = {'avg_val_loss': avg_loss, 'avg_val_acc': acc}

        print('val_loss: ', avg_loss.item(), 'acc: ', acc)

        return {'val_loss': avg_loss, 'log': tensorboard_logs}


    def test_step(self, batch, batch_idx, optimizer_idx = 0):
        
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img)
        acc = self.acc_function(results, labels, self.params['batch_size'])
        

        loss = self.loss_function(results, labels,
                                            M_N = self.params['batch_size']/ self.num_val_imgs,
                                            optimizer_idx = optimizer_idx,
                                            batch_idx = batch_idx)
        self.save_results(torch.sigmoid(results), batch_idx)
        return {'loss': loss, 'acc': acc}

    def test_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        acc = torch.stack([x['acc'] for x in outputs]).mean()

        print('test_loss', avg_loss, 'acc: ', acc)

        return {'test_loss': avg_loss, 'test_acc': acc}

    def configure_optimizers(self):
        return Adam(
            self.parameters(), 
            lr=self.params['LR'],
            betas = (0.9, 0.999)
        )

    def save_results(self, score, batch_idx):
        file_name = 'IDC_results.csv'
        if batch_idx == 0:
            f = open(file_name, 'w')
            result_writer = csv.writer(f)
            header = ['score']
            result_writer.writerow(header)
        else: 
            f = open(file_name, 'a')
            result_writer = csv.writer(f)

        for i, s in enumerate(score.tolist()):
            result_writer.writerow(s)
        f.close()

    @data_loader
    def train_dataloader(self):
        print('start train data loading')
       
        root = os.path.join(self.params['data_path'], self.params['dataset'] )
        dataset = IDC_Dataset(root = root, transform=self.data_transforms(), train = True)
        self.num_train_imgs = len(dataset)
        train_loader = DataLoader(dataset,
                                batch_size=self.params['batch_size'],
                                shuffle=True,
                                drop_last=True)

        print('end train data loading')
        return train_loader


    @data_loader
    def val_dataloader(self):
        root = os.path.join(self.params['data_path'], self.params['dataset'] )
        val_data = IDC_Dataset(root = root, transform=self.data_transforms(), train = False)
        self.num_val_imgs = len(val_data)
        val_loader = DataLoader(val_data,
                                batch_size=self.params['batch_size'],
                                shuffle=True,
                                drop_last=True)

        return val_loader
        
    
    @data_loader
    def test_dataloader(self):
        root = os.path.join(self.params['data_path'], self.params['dataset'] )
        test_data = IDC_Dataset(root = root, transform=self.data_transforms(), train = False)
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


if __name__ == "__main__":
    model = IDCPredictor()

    model.freeze()

    trainer = Trainer(gpus=0, max_epochs = 1, 
        early_stop_callback = False, 
        logger= False, # whether disable logs
        checkpoint_callback=False,
        show_progress_bar= True,
        weights_summary=None
        # reload_dataloaders_every_epoch=True # enable data loader switch between epoches
        )
    trainer.fit(model)
    # model.unfreeze()
    # trainer.fit(model)

    trainer.test(model)
