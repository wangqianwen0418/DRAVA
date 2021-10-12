import math
import torch
import numpy as np
import os
from torch import optim
from models import BaseVAE
from models.types_ import *
from utils import data_loader
import pytorch_lightning as pl
from torchvision import transforms
import torchvision.utils as vutils
from torchvision.datasets import CelebA
from torch.utils.data import Dataset, DataLoader
import json


class CustomTensorDataset(Dataset):
    def __init__(self, data_tensor):
        self.data_tensor = data_tensor

    def __getitem__(self, index):
        return self.data_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)

# extend the pytorch lightning module
class VAEModule(pl.LightningModule):

    def __init__(self,
                 vae_model: BaseVAE,
                 params: dict) -> None:
        super(VAEModule, self).__init__()

        self.model = vae_model
        self.params = params


        self.curr_device = torch.cuda.current_device()
        self.hold_graph = False

        self.bin_size = 11
        self.latent_hist = torch.zeros(self.model.latent_dim, self.bin_size).to(self.curr_device)
        try:
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass

    def is_np_dataset(self, dataset_name):
        # numpy datasets have different data loaders
        if 'sunspot' in dataset_name or dataset_name in ['dsprites', 'HFFc6_ATAC', 'ENCFF158GBQ']:
            return True
        else:
            return False
    
    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)


    def training_step(self, batch, batch_idx, optimizer_idx = 0):
        if self.is_np_dataset(self.params['dataset']):
            real_img = batch
            real_img = real_img.float()
            labels = '' # dummay labels for no-label dataset
        else:
            real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels = labels)
        train_loss = self.model.loss_function(*results,
                                              M_N = self.params['batch_size']/ self.num_train_imgs,
                                              optimizer_idx=optimizer_idx,
                                              batch_idx = batch_idx)

        self.logger.experiment.log({key: val.item() for key, val in train_loss.items()})

        return train_loss

    def training_end(self, outputs):
        
        return {'loss': outputs['loss']}

    def validation_step(self, batch, batch_idx, optimizer_idx = 0):
        if self.is_np_dataset(self.params['dataset']):
            real_img = batch
            real_img = real_img.float()
            labels = '' # dummay labels for no-label dataset
        else:
            real_img, labels = batch
        self.curr_device = real_img.device
        results = self.forward(real_img, labels = labels)
        val_loss = self.model.loss_function(*results,
                                            M_N = self.params['batch_size']/ self.num_val_imgs,
                                            optimizer_idx = optimizer_idx,
                                            batch_idx = batch_idx)

        return val_loss

    def validation_end(self, outputs):
        
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        tensorboard_logs = {'avg_val_loss': avg_loss}
        
        self.save_simu_images()
        self.save_paired_samples()

        print('val_loss: ', avg_loss.item())

        return {'val_loss': avg_loss, 'log': tensorboard_logs}


    def test_step(self, batch, batch_idx, optimizer_idx = 0):
        if self.is_np_dataset(self.params['dataset']):
            real_img = batch
            real_img = real_img.float()
            labels = '' # dummy labels
        else:
            real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels = labels)
        loss = self.model.loss_function(*results,
                                            M_N = self.params['batch_size']/ self.num_val_imgs,
                                            optimizer_idx = optimizer_idx,
                                            batch_idx = batch_idx)

        self.count_latent_dist(batch)
        return loss

    def test_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

        self.save_sample_dist(save_vector=True) 
        self.save_latent_hist()
        self.save_simu_images()
        print('test_loss', avg_loss)

        return {'test_loss': avg_loss}

    def count_latent_dist(self, batch):
        """
        count the value distribution at each latent dimension
        """
        
        if self.is_np_dataset(self.params['dataset']):
            real_img = batch
            real_img = real_img.float()
            labels = '' # dummy labels
        else:
            real_img, labels = batch
        self.curr_device = real_img.device

        [recons, test_input, mu, log_var] = self.forward(real_img, labels = labels)
        latent_hist = [ torch.histc( mu[:, i], bins= self.bin_size, min=-3, max= 3) for i in range(self.model.latent_dim)]
        latent_hist = torch.stack(latent_hist) # latent_dim * bin_size
        latent_hist = latent_hist.to(self.curr_device)
        self.latent_hist = latent_hist + self.latent_hist

    def save_sample_dist(self, save_vector=False):
        """
        save real input samples to demonstrate the latent dim value distribution 
        """
        if self.is_np_dataset(self.params['dataset']):
            test_input = next(iter(self.sample_dataloader))
            test_input = test_input.float()
            test_label = '' # dummy labels
        else:   
            test_input, test_label = next(iter(self.sample_dataloader))
            test_label = test_label.to(self.curr_device)
        test_input = test_input.to(self.curr_device)
        

        [recons, test_input, mu, log_var] = self.forward(test_input, labels = test_label)
        with open(f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/sample.json", 'w') as f:
            json.dump(mu.tolist(), f)

        filepath = f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/imgs"
        if not(os.path.isdir(filepath)):
            os.mkdir(filepath)
        vutils.save_image(test_input.data,
                            f"{filepath}/real_samples.png",
                            normalize=True,
                            nrow=10)

        if save_vector:
            torch.save(mu, f"{filepath}/real_samples_vector.pt")

    def save_latent_hist(self):
        """
        save the value distribution of training images in each hidden dimension
        """   
        with open(f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/hist.json", 'w') as f:
            json.dump(self.latent_hist.tolist(), f)


    def save_simu_images(self):
        """
        return an image grid,
        each row is a hidden dimension, 
        all images in this row have same values for other dims but differnt values at this dim  
        """
        nrow = 11 # number of images at each row

        z = []
        for i in range(self.model.latent_dim):
            # z_ = torch.randn( self.model.latent_dim)
            z_ = torch.zeros( self.model.latent_dim)
            z_ = [z_ for i in range(nrow)]
            z_ = torch.stack(z_, dim =0)
            mask = torch.tensor([j for j in range(nrow)])
            z_[mask, i] = torch.tensor([- 3 + j/(nrow-1)* 6 for j in range(nrow)]).float()

            z.append(z_)
        z = torch.stack(z)
        z = z.to(self.curr_device)

        samples = self.model.decode(z)

        
        filepath = f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/imgs"
        if not(os.path.isdir(filepath)):
            os.mkdir(filepath)
        vutils.save_image(samples.cpu().data,
                            f"{filepath}/{self.logger.name}_simu_samples_{self.current_epoch}.png",
                            normalize=True,
                            nrow=nrow)
   
    
    def save_paired_samples(self):
        """
        run at the end of each epoch,
        save input sample images and their reconstructed images
        """
        if self.is_np_dataset(self.params['dataset']):
            test_input = next(iter(self.sample_dataloader))
            test_input = test_input.float()
            test_label = '' # dummy labels
        else:   
            test_input, test_label = next(iter(self.sample_dataloader))
            test_label = test_label.to(self.curr_device)
        test_input = test_input.to(self.curr_device)
        
        recons = self.model.generate(test_input, labels = test_label)
        
        filepath = f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/imgs"
        if not(os.path.isdir(filepath)):
            os.mkdir(filepath)

        # input images
        vutils.save_image(test_input.data,
                          f"{filepath}/real_img_{self.logger.name}_{self.current_epoch}.png",
                          normalize=True,
                          nrow=12)
        
        # reconstructed images
        vutils.save_image(recons.data,
                          f"{filepath}/recons_{self.logger.name}_{self.current_epoch}.png",
                          normalize=True,
                          nrow=12)

        try:
            self.save_simu_images()
        except:
            pass


        del test_input, recons #, samples


    def configure_optimizers(self):

        optims = []
        scheds = []

        if self.params['optimizer'] == 'adagrad':
            optimizer = optim.Adagrad(self.model.parameters(), lr=self.params['LR'],
                                weight_decay=self.params['weight_decay'])
        
        else:
            optimizer = optim.Adam(self.model.parameters(),
                                lr=self.params['LR'],
                                weight_decay=self.params['weight_decay'])
        optims.append(optimizer)
        # Check if more than 1 optimizer is required (Used for adversarial training)
        try:
            if self.params['LR_2'] is not None:
                optimizer2 = optim.Adam(getattr(self.model,self.params['submodel']).parameters(),
                                        lr=self.params['LR_2'])
                optims.append(optimizer2)
        except:
            pass

        try:
            if self.params['scheduler_gamma'] is not None:
                scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
                                                             gamma = self.params['scheduler_gamma'])
                scheds.append(scheduler)

                # Check if another scheduler is required for the second optimizer
                try:
                    if self.params['scheduler_gamma_2'] is not None:
                        scheduler2 = optim.lr_scheduler.ExponentialLR(optims[1],
                                                                      gamma = self.params['scheduler_gamma_2'])
                        scheds.append(scheduler2)
                except:
                    pass
                return optims, scheds
        except:
            return optims

    @data_loader
    def train_dataloader(self):

        if self.params['dataset'] == 'celeba':
            transform = self.data_transforms()
            dataset = CelebA(root = self.params['data_path'],
                             split = "train",
                             transform=transform,
                             download=False)
            self.num_train_imgs = len(dataset)
            return DataLoader(dataset,
                            batch_size= self.params['batch_size'],
                            shuffle = True,
                            drop_last=True)

        elif self.is_np_dataset(self.params['dataset']):
            print('start train data loading')
            root = os.path.join(self.params['data_path'], f"{self.params['dataset']}.npz")
            if not os.path.exists(root):
                import subprocess
                print('Now download dsprites-dataset')
                subprocess.call(['./download_dsprites.sh'])
                print('Finished')
            data = np.load(root, encoding='bytes')
            tensor = torch.from_numpy(data['imgs']).unsqueeze(1)
            print('train data loading')
            train_kwargs = {'data_tensor':tensor}
            dset = CustomTensorDataset
            train_data = dset(**train_kwargs)
            self.num_train_imgs = len(train_data)
            train_loader = DataLoader(train_data,
                                    batch_size=self.params['batch_size'],
                                    shuffle=True,
                                    drop_last=True)
            print('end train data loading')
            return train_loader
        else:
            raise ValueError('Undefined dataset type')


    @data_loader
    def val_dataloader(self):

        if self.params['dataset'] == 'celeba':
            transform = self.data_transforms()
            self.sample_dataloader =  DataLoader(CelebA(root = self.params['data_path'],
                                                        split = "test",
                                                        transform=transform,
                                                        download=False),
                                                 batch_size= 144,
                                                 shuffle = True,
                                                 drop_last=True)
            self.num_val_imgs = len(self.sample_dataloader)
            return self.sample_dataloader
        elif self.is_np_dataset(self.params['dataset']):
            print('start val data loading')
            # since the two datasets are small, use the train data loader for val
            self.sample_dataloader = self.train_dataloader()
            self.num_val_imgs = self.num_train_imgs
            return self.train_dataloader()
        else:
            raise ValueError('Undefined dataset type')
    
    @data_loader
    def test_dataloader(self):
        if self.params['dataset'] == 'celeba':
            transform = self.test_data_transforms()
            self.test_sample_dataloader =  DataLoader(CelebA(root = self.params['data_path'],
                                                        split = "test",
                                                        transform=transform,
                                                        download=False),
                                                 batch_size= 144,
                                                 shuffle = False,
                                                 drop_last=False)
            self.num_test_imgs = len(self.test_sample_dataloader)
            return self.test_sample_dataloader

        elif self.is_np_dataset(self.params['dataset']):
            # since the two datasets are small, use the train data loader for test
            return self.val_dataloader()
        else:
            raise ValueError('Undefined dataset type')


    def data_transforms(self):

        SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
        SetScale = transforms.Lambda(lambda X: X/X.sum(0).expand_as(X))

        if self.params['dataset'] == 'celeba':
            transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                            transforms.CenterCrop(148),
                                            transforms.Resize(self.params['img_size']),
                                            transforms.ToTensor(),
                                            SetRange])
        
        else:
            raise ValueError('Undefined dataset type')
        return transform

    def test_data_transforms(self):
        SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
        if self.params['dataset'] == 'celeba':
            transform = transforms.Compose([transforms.CenterCrop(148),
                                            transforms.Resize(self.params['img_size']),
                                            transforms.ToTensor(),
                                            SetRange])
        else:
            raise ValueError('Undefined dataset type')
        return transform