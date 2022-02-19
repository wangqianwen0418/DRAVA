import math
import numpy as np
import os
from pathlib import Path
import pandas as pd
import json
import csv


import torch
from torch import optim
import pytorch_lightning as pl
from torchvision import transforms
import torchvision.utils as vutils
from torchvision.datasets import CelebA
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import SGD, Adam, Adagrad
from PIL import Image

from models import BaseVAE
from models.types_ import *
from utils import data_loader


class CustomTensorDataset(Dataset):
    def __init__(self, data_tensor, labels):
        self.data_tensor = data_tensor
        self.labels = labels

    def __getitem__(self, index):
        return self.data_tensor[index], self.labels[index]

    def __len__(self):
        return self.data_tensor.size(0)




class CustomImageDataset(Dataset):
    def __init__(self, root, transform=None, target_transform=None, chr=None):
        df = pd.read_csv(os.path.join(root, 'label.csv'))
        if (chr!= None):
            self.img_labels = df[df['img'].str.contains(chr)] #e.g., chr = 'chr5'
        else:
            self.img_labels = df
        self.img_dir = root
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, f'{self.img_labels.iloc[idx, 0]}.jpg')
        image = Image.open(img_path).convert('L')
        # 
        label = self.img_labels.iloc[idx].copy()
        try:
            label[0] = float(label[0].split(':')[0].replace('chr', '')) # get the CHR number from the jpg name
        except Exception:
            label[0] = 7 # chr 7 dataset
        label = label.to_numpy(dtype='float')

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

# extend the pytorch lightning module
class VAEModule(pl.LightningModule):

    def __init__(self,
                 vae_model: BaseVAE,
                 params: dict) -> None:
        super(VAEModule, self).__init__()

        self.model = vae_model
        self.params = params

        if torch.cuda.is_available():
            device = torch.cuda.current_device()
        else:
            device = torch.device("cpu")
        self.curr_device = device
        
        self.hold_graph = False

        self.bin_num = 11
        try:
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass

        # min and max value for each latent dim, used to generate simu images
        # to avoid long tail, use the top k value as the max value, least k value as the min value
        K = 5
        self.z_range = [ [ [math.inf for _ in range(K) ], [-math.inf for _ in range(K)]  ] for _ in range(self.model.latent_dim) ]

    @property
    def logger_folder(self):
        return f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/"

    def is_tensor_dataset(self, dataset_name):
        # numpy datasets have different data loaders
        if 'sunspot' in dataset_name or dataset_name in ['dsprites', 'HFFc6_ATAC_chr7', 'HFFc6_ATAC_chr1-8', 'ENCFF158GBQ']:
            return True
        else:
            return False

    def is_hic_dataset(self, dataset_name):
        # whether to use custom image data loader for hi c data
        if dataset_name in ['TAD_GM12878', 'TAD_HFFc6_chr7_10k','TAD_HFFc6_10k_chr1-5']:
            return True
        else:
            return False
    
    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)


    def training_step(self, batch, batch_idx, optimizer_idx = 0):

        
        real_img, labels = batch

        if self.is_tensor_dataset(self.params['dataset']):
            real_img = real_img.float()

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
        real_img, labels = batch

        if self.is_tensor_dataset(self.params['dataset']):           
            real_img = real_img.float()

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
        
        real_img, labels = batch

        if self.is_tensor_dataset(self.params['dataset']):            
            real_img = real_img.float()
        
        self.curr_device = real_img.device

        results = self.forward(real_img, labels = labels)
        mu = results[2]

        loss = self.model.loss_function(*results,
                                            M_N = self.params['batch_size']/ self.num_val_imgs,
                                            optimizer_idx = optimizer_idx,
                                            batch_idx = batch_idx)

        # save latent vectors of samples in this batch
        self.save_results(mu, labels, batch_idx)
        return loss

    def test_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()


        # save z range
        f = open(os.path.join(self.logger_folder, 'results/', 'z_range.json'), 'w')     
        ranges = []
        for dim in self.z_range:
            row = [ dim[0][-1], dim[1][0] ]
            ranges.append(row)
        json.dump(ranges, f)
        f.close()

        # save image reconstruction space
        self.save_simu_images(as_individual=True, ranges = ranges, is_test = True)
        print('test_loss', avg_loss)

        return {'test_loss': avg_loss}

    def save_results(self, mu, labels, batch_idx):
        """
        save results in a csw file, columns are [labels] + [latent_z]
        """
        filepath = f"{self.logger_folder}/results/"
        if batch_idx == 0:
            Path(filepath).mkdir(parents=True, exist_ok = True)
            f = open(os.path.join(filepath, 'results.csv'), 'w')
            result_writer = csv.writer(f)

            if self.params['dataset'] == 'celeba':
                header = ['z']
            else: 
                header = ['chr', 'start', 'end', 'level', 'mean', 'score'][0: len(labels[0])] + ['z']
            result_writer.writerow(header)
        else: 
            f = open(os.path.join(filepath, 'results.csv'), 'a')
            result_writer = csv.writer(f)


        for i, m in enumerate(mu.tolist()):
            if self.params['dataset'] == 'celeba':
                row = [','.join([str(d) for d in m])]
            else:
                row = labels[i].tolist() + [','.join([str(d) for d in m])]

            result_writer.writerow(row)

            # get range z
            for j, d in enumerate(m):
                if d < self.z_range[j][0][-1]:
                    self.z_range[j][0][-1] = d
                    self.z_range[j][0].sort()

                if d > self.z_range[j][1][0]:
                    self.z_range[j][1][0] = d
                    self.z_range[j][1].sort()

        f.close()

    def save_simu_images(self, as_individual=False, ranges = [], is_test=False):
        """
        return an image grid,
        each row is a hidden dimension, 
        all images in this row have same values for other dims but differnt values at this dim  
        """

        z = []
        for i in range(self.model.latent_dim):
            baseline = torch.randn( self.model.latent_dim) - 0.5
            # baseline = torch.zeros( self.model.latent_dim)
            # baseline = torch.ones( self.model.latent_dim) 
            # baseline = torch.randn( self.model.latent_dim)
            if len(ranges)>0:
                baseline = torch.tensor( 
                    [(ranges[i][0] + ranges[i][1])/2 for i in range(self.model.latent_dim)]
                    )
            # baseline = torch.tensor([0.5784032344818115,0.1713341921567917,-0.27981624007225037,-0.4180270731449127,0.9767476916313171,-0.7862354516983032,0.7032433152198792,0.7099565863609314])
            baseline = torch.tensor([1.3912068605422974,1.3093589544296265,-1.4369394779205322,2.921229362487793,1.7272869348526,-1.0809800624847412])
            z_ = [baseline for _ in range(self.bin_num)]
            z_ = torch.stack(z_, dim =0)
            mask = torch.tensor([j for j in range(self.bin_num)])

            if len(ranges) == 0:
                z_min = -3
                z_max = 3
            else:
                z_min = ranges[i][0]
                z_max = ranges[i][1]    
            z_[mask, i] = torch.tensor(
                [z_min + j/(self.bin_num-1)* (z_max - z_min) for j in range(self.bin_num)]
            ).float()

            z.append(z_)
        z = torch.stack(z)
        z = z.to(self.curr_device) # the shape of z: [ latent_dim * bin_size, latent_dim ]

        recons = self.model.decode(z)

        filepath = f"{self.logger_folder}/imgs"
        Path(filepath).mkdir(parents=True, exist_ok=True)

        if self.is_tensor_dataset(self.params['dataset']):
            recons_imgs = (recons.cpu().data>0.5).float() # so that the simulated images have only white and black and no gray
        else:
            recons_imgs = recons.cpu().data
        
        if as_individual:
            Path(f"{self.logger_folder}/results/simu").mkdir(parents=True, exist_ok=True)
            img_idx = 0
            for img in recons_imgs:
                q, mod = divmod(img_idx, self.bin_num)
                vutils.save_image(img, f"{self.logger_folder}/results/simu/{q}_{mod}.png",)
                img_idx += 1
        
        if is_test:
            save_path = f"{self.logger_folder}/results/simu.png"
        else:
            save_path = f"{filepath}/{self.logger.name}_simu_samples_{self.current_epoch}.png"
        
        vutils.save_image(recons_imgs,
                            save_path,
                            normalize=True,
                            nrow=self.bin_num)

    def get_simu_images(self, dimIndex, baseline = [], z_range = []):
        """
        return an image grid,
        each row is a hidden dimension, 
        all images in this row have same values for other dims but differnt values at this dim  
        """


        if len(baseline) > 0:
            baseline = torch.tensor(baseline)
        elif len(z_range)>0:
            baseline = torch.tensor( 
                [(z_range[0] + z_range[1])/2 for i in range(self.model.latent_dim)]
                )
        else: baseline = torch.randn( self.model.latent_dim) - 0.5
        
        z = [baseline for _ in range(self.bin_num)]
        z = torch.stack(z, dim =0)
        mask = torch.tensor([j for j in range(self.bin_num)])

        if len(z_range) == 0:
            z_min = -3
            z_max = 3
        else:
            z_min = z_range[0]
            z_max = z_range[1]    
        z[mask, dimIndex] = torch.tensor(
            [z_min + j/(self.bin_num-1)* (z_max - z_min) for j in range(self.bin_num)]
        ).float()
        recons = self.model.decode(z)

        if self.is_tensor_dataset(self.params['dataset']):
            recons_imgs = (recons.cpu().data>0.5).float() # so that the simulated images have only white and black and no gray
        else:
            recons_imgs = recons.cpu().data
        
        
        # vutils.save_image(recons_imgs,
        #                     './test_simu.png',
        #                     normalize=True,
        #                     nrow=self.bin_num)

        return recons_imgs
    
    def save_paired_samples(self):
        """
        run at the end of each epoch,
        save input sample images and their reconstructed images
        """
        test_input, test_label = next(iter(self.sample_dataloader))
        if self.is_tensor_dataset(self.params['dataset']):          
            test_input = test_input.float()
            
        test_label = test_label.to(self.curr_device)
        test_input = test_input.to(self.curr_device)
        
        recons = self.model.generate(test_input, labels = test_label)
        
        filepath = f"{self.logger_folder}/imgs"
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

        optimizer_dict = {'adagrad': Adagrad, 'adam': Adam, 'sgd': SGD}

        assert self.params['optimizer'] in [*optimizer_dict], f'only support {[*optimizer_dict]} as optimizers'


        optimizer = optimizer_dict[self.params['optimizer']](
                                self.model.parameters(), 
                                lr=self.params['LR'],
                                weight_decay=self.params['weight_decay']
                                )
        
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
            else:
                # if no schedular gama, reduce LR by factor (0.1) when a metric has stopped improving
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optims[0], factor = 0.1, patience = 10, verbose = True)
                scheds.append(scheduler)
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

        elif self.is_hic_dataset(self.params['dataset']):
            root = os.path.join(self.params['data_path'], self.params['dataset'] )
            dataset = CustomImageDataset(root = root, transform=self.data_transforms())
            self.num_train_imgs = len(dataset)

            if 'weighted_sampler' in self.params and self.params['weighted_sampler']:
                # resampling training data based on the inital size
                weights = torch.tensor([1, 3]) # upsampling by 3 if the size if larger than 20 * 10K

                def get_weight(labels, sample_index, weights):
                    label = labels.iloc[sample_index]
                    return weights[ 1 if (label['end'] - label['start'] > 24) else 0 ]
                size_sampler = WeightedRandomSampler(
                    weights = [get_weight(dataset.img_labels, i, weights) for i in range(self.num_train_imgs)],
                    num_samples = self.num_train_imgs,
                    replacement = True
                )
                
                return DataLoader(dataset,
                                batch_size= self.params['batch_size'],
                                # shuffle = True, # no shuffle is sample is used
                                sampler = size_sampler,
                                drop_last=True)
            else:
                return DataLoader(dataset,
                                batch_size= self.params['batch_size'],
                                shuffle = True, 
                                drop_last=True)
        
        elif self.is_tensor_dataset(self.params['dataset']):
            print('start train data loading')
            root = os.path.join(self.params['data_path'], f"{self.params['dataset']}.npz")
            if not os.path.exists(root):
                import subprocess
                print('Now download dsprites-dataset')
                subprocess.call(['./download_dsprites.sh'])
                print('Finished')
            data = np.load(root, encoding='bytes')
            tensor = torch.from_numpy(data['imgs']).unsqueeze(1) # unsequeeze reshape data from [x, 64, 64] to [x, 1, 64, 64]
            labels = torch.from_numpy(data['labels'])

            # transform = self.data_transforms()
            # tensor = transform(tensor)

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
        elif self.is_hic_dataset(self.params['dataset']) or self.is_tensor_dataset(self.params['dataset']):
            print('start val data loading')
            # since the two datasets are small, use the train data loader for val
            self.sample_dataloader = self.train_dataloader()
            self.num_val_imgs = self.num_train_imgs
            return self.train_dataloader()
        else:
            raise ValueError('Undefined dataset type')
    
    @data_loader
    def test_dataloader(self):
        print('loading test data')
        if self.params['dataset'] == 'celeba':
            transform = self.test_data_transforms()
            self.test_sample_dataloader =  DataLoader(CelebA(root = self.params['data_path'],
                                                        split = "train",
                                                        transform=transform,
                                                        download=False),
                                                 batch_size= 144,
                                                 shuffle = False,
                                                 drop_last=False)
            self.num_test_imgs = len(self.test_sample_dataloader)
            return self.test_sample_dataloader

        elif self.is_hic_dataset(self.params['dataset']):

            root = os.path.join(self.params['data_path'], self.params['dataset'] )
            dataset = CustomImageDataset(root = root, transform=self.test_data_transforms())
            self.test_sample_dataloader = DataLoader(dataset,
                            batch_size= self.params['batch_size'],
                            shuffle = False,
                            drop_last = False)

            self.num_test_imgs = len(self.test_sample_dataloader)
            return self.test_sample_dataloader

        elif self.is_tensor_dataset(self.params['dataset']):
            root = os.path.join(self.params['data_path'], f"{self.params['dataset']}.npz")
            if not os.path.exists(root):
                import subprocess
                print('Now download dsprites-dataset')
                subprocess.call(['./download_dsprites.sh'])
                print('Finished')
            data = np.load(root, encoding='bytes')
            tensor = torch.from_numpy(data['imgs']).unsqueeze(1) # unsequeeze reshape data from [x, 64, 64] to [x, 1, 64, 64]
            labels = torch.from_numpy(data['labels'])

            # transform = self.data_transforms()
            # tensor = transform(tensor)

            train_kwargs = {'data_tensor':tensor, 'labels': labels}
            dset = CustomTensorDataset
            train_data = dset(**train_kwargs)
            self.test_sample_dataloader = DataLoader(train_data,
                                    batch_size=self.params['batch_size'],
                                    shuffle=False,
                                    drop_last=False)
            
            self.num_test_imgs = len(self.test_sample_dataloader)
            return self.test_sample_dataloader
        else:
            raise ValueError('Undefined dataset type')


    def data_transforms(self):
        if self.is_hic_dataset(self.params['dataset']):
            SetRange = transforms.Lambda(lambda X: 2 * X - 1.) # [0,1] to [-1, 1]
            transform = transforms.Compose([transforms.Resize(self.params['img_size'], Image.NEAREST),
                                            transforms.RandomApply([transforms.RandomHorizontalFlip(1), transforms.RandomVerticalFlip(1)], 0.5),
                                            transforms.ToTensor(),
                                            SetRange])

        elif self.params['dataset'] == 'celeba':
            SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
            transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                            transforms.CenterCrop(148),
                                            transforms.Resize(self.params['img_size']),
                                            transforms.ToTensor(),
                                            SetRange])
        
        else:
            # do not use transforms for tensor datasets for now
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor()
                ])
        return transform

    def test_data_transforms(self):
        if self.params['dataset'] == 'celeba':
            SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
            transform = transforms.Compose([transforms.CenterCrop(148),
                                            transforms.Resize(self.params['img_size']),
                                            transforms.ToTensor(),
                                            SetRange])
        elif self.is_hic_dataset(self.params['dataset']):
            SetRange = transforms.Lambda(lambda X: 2 * X - 1.) # [0,1] to [-1, 1]
            transform = transforms.Compose([transforms.Resize(self.params['img_size'], Image.NEAREST),
                                            transforms.ToTensor(),
                                            SetRange])
        else:
            # 
            # do not use transforms for tensor datasets for now
            transform = transforms.Compose([
                # transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor()
                ])
        return transform