import math
import numpy as np
import os
from pathlib import Path
import json
import csv

import torch
from torch import optim
import pytorch_lightning as pl
from torchvision import transforms
import torchvision.utils as vutils
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from torch.optim import SGD, Adam, Adagrad
import torch.nn.functional as F

from PIL import Image

from models import BaseVAE
from models.types_ import *
from utils import data_loader, drawMasks
from dataloaders import CustomTensorDataset, CodeX_Dataset, CodeX_Landmark_Dataset, HiC_Dataset, IDC_Dataset


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
        self.concept_array = []
        self.z_range = [[[math.inf for _ in range(
            K)], [-math.inf for _ in range(K)]] for _ in range(self.model.latent_dim)]

    @property
    def logger_folder(self):
        return f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/"

    def is_tensor_dataset(self, dataset_name):
        # numpy datasets have different data loaders
        if 'sunspot' in dataset_name or dataset_name in ['dsprites', 'dsprites_test', 'HFFc6_ATAC_chr7', 'HFFc6_ATAC_chr1-8', 'ENCFF158GBQ']:
            return True
        else:
            return False

    def is_hic_dataset(self, dataset_name):
        # whether to use custom image data loader for hi c data
        if dataset_name in ['TAD_GM12878', 'TAD_HFFc6_chr7_10k', 'TAD_HFFc6_10k_chr1-5']:
            return True
        else:
            return False

    def is_IDC_dataset(self, dataset_name):
        # whether to use custom image data loader for hi c data
        if dataset_name in ['IDC_regular_ps50_idx5']:
            return True
        else:
            return False

    def is_genomic_dataset(self, dataset_name):
        # whether to use custom image data loader for hi c data
        if dataset_name in ['HFFc6_ATAC_chr7', 'HFFc6_ATAC_chr1-8', 'ENCFF158GBQ'] or self.is_hic_dataset(dataset_name):
            return True
        else:
            return False

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx=0):

        real_img, labels = batch

        if self.is_tensor_dataset(self.params['dataset']):
            real_img = real_img.float()

        self.curr_device = real_img.device

        results = self.forward(real_img, labels=labels)
        train_loss = self.model.loss_function(*results,
                                              M_N=self.params['batch_size'] /
                                              self.num_train_imgs,
                                              optimizer_idx=optimizer_idx,
                                              batch_idx=batch_idx)

        self.logger.experiment.log({key: val.item()
                                    for key, val in train_loss.items()})

        return train_loss

    def training_end(self, outputs):

        return {'loss': outputs['loss']}

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        real_img, labels = batch

        if self.is_tensor_dataset(self.params['dataset']):
            real_img = real_img.float()

        self.curr_device = real_img.device
        results = self.forward(real_img, labels=labels)
        val_loss = self.model.loss_function(*results,
                                            M_N=self.params['batch_size'] /
                                            self.num_val_imgs,
                                            optimizer_idx=optimizer_idx,
                                            batch_idx=batch_idx)

        return val_loss

    def validation_end(self, outputs):

        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        tensorboard_logs = {'avg_val_loss': avg_loss}

        self.save_simu_images()
        self.save_paired_samples()

        print('val_loss: ', avg_loss.item())

        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def test_step(self, batch, batch_idx, optimizer_idx=0):

        real_img, labels = batch

        if self.is_tensor_dataset(self.params['dataset']):
            real_img = real_img.float()

        self.curr_device = real_img.device

        results = self.forward(real_img, labels=labels)
        mu = results[2]
        log_var = results[3]
        std = torch.exp(0.5 * log_var)

        loss = self.model.loss_function(*results,
                                        M_N=self.params['batch_size'] /
                                        self.num_val_imgs,
                                        optimizer_idx=optimizer_idx,
                                        batch_idx=batch_idx)

        # save output for concept adaptor
        concept_in = self.concept_encoder(real_img)
        self.concept_array.append([concept_in, mu, std, labels])

        # save latent vectors of samples in this batch
        self.save_results(mu, loss['Reconstruction_Loss'], labels, batch_idx)
        return loss

    def test_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

        # save concept arrays
        concepts_in = torch.cat([i[0] for i in self.concept_array], 0)
        concepts_in = concepts_in.cpu().detach().numpy()
        mu = torch.cat([i[1] for i in self.concept_array], 0)
        mu = mu.cpu().detach().numpy()
        std = torch.cat([i[2] for i in self.concept_array], 0)
        std = std.cpu().detach().numpy()
        labels = torch.cat([i[3] for i in self.concept_array], 0)
        labels = labels.cpu().detach().numpy()
        np.savez(f'./data/{self.params["dataset"]}_concepts.npz',
                 x=concepts_in, y=mu, std=std, gt=labels)

        # save z range
        f = open(os.path.join(self.logger_folder,
                              'results/', 'z_range.json'), 'w')
        ranges = []
        for dim in self.z_range:
            row = [dim[0][-1], dim[1][0]]
            ranges.append(row)
        json.dump(ranges, f)
        f.close()

        # save image reconstruction space
        self.save_simu_images(as_individual=True, ranges=ranges, is_test=True)
        print('test_loss', avg_loss)

        return {'test_loss': avg_loss}

    def concept_encoder(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """

        # output of th last conv layer will be used to learn the concept

        # # for the beta_vae_conv model
        # concept_in = self.model.encoder(input)

        # for the beta_vae_conv2 model
        if 'dsprites' in self.params['dataset']:
            concept_in = self.model.encoder[:8](input)
        else:
            concept_in = self.model.encoder(input)

        return concept_in

    def save_results(self, mu, recons_loss, labels, batch_idx):
        """
        save results in a csw file, columns are [labels] + [latent_z]
        """
        filepath = f"{self.logger_folder}/results/"
        if batch_idx == 0:
            Path(filepath).mkdir(parents=True, exist_ok=True)
            f = open(os.path.join(filepath, 'results.csv'), 'w')
            result_writer = csv.writer(f)

            if self.is_genomic_dataset(self.params['dataset']):
                header = ['chr', 'start', 'end', 'level', 'mean',
                          'score'][0: len(labels[0])] + ['z', 'recons_loss']
            else:
                header = ['z', 'recons_loss']

            result_writer.writerow(header)
        else:
            f = open(os.path.join(filepath, 'results.csv'), 'a')
            result_writer = csv.writer(f)

        recons_loss = recons_loss.tolist()
        for i, m in enumerate(mu.tolist()):

            if self.is_genomic_dataset(self.params['dataset']):
                row = labels[i].tolist() + [','.join([str(d)
                                                      for d in m]), recons_loss[i]]
            else:
                row = [','.join([str(d) for d in m]), recons_loss[i]]

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

    def save_simu_images(self, as_individual=False, ranges=[], is_test=False):
        """
        return an image grid,
        each row is a hidden dimension, 
        all images in this row have same values for other dims but differnt values at this dim  
        """

        z = []
        for i in range(self.model.latent_dim):
            if len(ranges) > 0:
                baseline = torch.tensor(
                    [(ranges[i][0] + ranges[i][1]) /
                     2 for i in range(self.model.latent_dim)]
                )
            else:
                baseline = torch.randn(self.model.latent_dim) - 0.5
            z_ = [baseline for _ in range(self.bin_num)]
            z_ = torch.stack(z_, dim=0)
            mask = torch.tensor([j for j in range(self.bin_num)])

            if len(ranges) == 0:
                z_min = -3
                z_max = 3
            else:
                z_min = ranges[i][0]
                z_max = ranges[i][1]
            z_[mask, i] = torch.tensor(
                [z_min + j/(self.bin_num-1) * (z_max - z_min)
                 for j in range(self.bin_num)]
            ).float()

            z.append(z_)
        z = torch.stack(z)
        # the shape of z: [ latent_dim * bin_size, latent_dim ]
        z = z.to(self.curr_device)

        recons = self.model.decode(z)

        filepath = f"{self.logger_folder}/imgs"
        Path(filepath).mkdir(parents=True, exist_ok=True)

        # if self.is_tensor_dataset(self.params['dataset']):
        #     recons_imgs = F.sigmoid (recons).cpu().data
        if self.is_tensor_dataset(self.params['dataset']):
            # so that the simulated images have only white and black and no gray
            recons_imgs = (recons.cpu().data > 0.5).float()
        else:
            recons_imgs = recons.cpu().data

        if 'codex' in self.params['dataset'] and 'num_cluster' not in self.params:
        # [TODO what is the best way to show multiplex images here?]
            recons_imgs = recons_imgs[:, 0:3, :, :]

        if as_individual:
            Path(
                f"{self.logger_folder}/results/simu").mkdir(parents=True, exist_ok=True)
            img_idx = 0
            for img in recons_imgs:
                q, mod = divmod(img_idx, self.bin_num)
                if 'codex' in self.params['dataset'] and 'num_cluster' in self.params:
                    drawMasks(
                        np.expand_dims(recons_imgs.numpy(), axis=0), 
                        figsize = 60, nrows = 1, save_path=f"{self.logger_folder}/results/simu/{q}_{mod}.png"
                        )
                else:
                    vutils.save_image(
                        img, f"{self.logger_folder}/results/simu/{q}_{mod}.png",)
                img_idx += 1

        if is_test:
            save_path = f"{self.logger_folder}/results/simu.png"
        else:
            save_path = f"{filepath}/{self.logger.name}_simu_samples_{self.current_epoch}.png"

        if 'codex' in self.params['dataset'] and 'num_cluster' in self.params:
            drawMasks(recons_imgs.numpy(), figsize = 60, nrows = self.bin_num, save_path=save_path)
        else:
            vutils.save_image(recons_imgs,
                            save_path,
                            normalize=True,
                            nrow=self.bin_num)

    def z2recons_sum(self, z):
        '''
        # feed for Shap to calculated z importances
        '''
        z = torch.tensor(z).float()
        recons = self.model.decode(z)

        return recons.view(recons.size(0), -1).sum(dim=1)

    def get_simu_images(self, dimIndex, baseline=[], z_range=[]):
        """
        Called by Flask Api to generate simu images
        return an image grid,
        each row is a hidden dimension, 
        all images in this row have same values for other dims but differnt values at this dim  
        @Return: images: numpy.array(), score: number
        """

        if len(baseline) > 0:
            baseline = torch.tensor(baseline)
        elif len(z_range) > 0:
            baseline = torch.tensor(
                [(z_range[0] + z_range[1])/2 for i in range(self.model.latent_dim)]
            )
        else:
            baseline = torch.randn(self.model.latent_dim) - 0.5

        z = [baseline for _ in range(self.bin_num)]
        z = torch.stack(z, dim=0)
        mask = torch.tensor([j for j in range(self.bin_num)])

        if len(z_range) == 0:
            z_min = -3
            z_max = 3
        else:
            z_min = z_range[0]
            z_max = z_range[1]
        z[mask, dimIndex] = torch.tensor(
            [z_min + j/(self.bin_num-1) * (z_max - z_min)
             for j in range(self.bin_num)]
        ).float()

        recons = self.model.decode(z)

        if self.is_tensor_dataset(self.params['dataset']):
            # so that the simulated images have only white and black and no gray
            recons = (recons.cpu().data > 0.5).float()
        else:
            recons = recons.cpu().data

        # the same normalization as torch.utils.save_image
        for t in recons:
            min = float(t.min())
            max = float(t.max())
            t.clamp_(min=min, max=max)
            t.add_(-min).div_(max - min + 1e-5)  # add 1e-5 in case min = max

        recons = recons.numpy()
        avg = recons.mean(axis=0)
        grad_score = [np.mean(np.abs(res-avg)) for res in recons]

        return recons, sum(grad_score)/len(grad_score)

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

        recons = self.model.generate(test_input, labels=test_label)
        # if self.is_tensor_dataset(self.params['dataset']):
        #     recons_imgs = F.sigmoid (recons).cpu().data
        if self.is_tensor_dataset(self.params['dataset']):
            # so that the simulated images have only white and black and no gray
            recons_imgs = (recons.cpu().data > 0.5).float()
        else:
            recons_imgs = recons.cpu().data

        filepath = f"{self.logger_folder}/imgs"
        if not(os.path.isdir(filepath)):
            os.mkdir(filepath)


        # input images
        input_save_path = f"{filepath}/real_img_{self.logger.name}_{self.current_epoch}.png"
        if 'codex' in self.params['dataset'] and 'num_cluster' in self.params:
            drawMasks(test_input.numpy(), figsize = 60, nrows = 12, save_path=input_save_path)        
        else:
            vutils.save_image(test_input.data[:, 0:3, :, :],  # [TODO what is the best way to show multiplex images here?]
                            input_save_path,
                            normalize=True,
                            nrow=12)

        # reconstructed images
        recons_save_path = f"{filepath}/recons_{self.logger.name}_{self.current_epoch}.png"
        if 'codex' in self.params['dataset'] and 'num_cluster' in self.params:
            drawMasks(recons_imgs.numpy(), figsize = 60, nrows = 12, save_path=recons_save_path)        
        else:
            vutils.save_image(recons_imgs[:, 0:3, :, :],  # [TODO what is the best way to show multiplex images here?]
                            recons_save_path,
                            normalize=True,
                            nrow=12)

        try:
            self.save_simu_images()
        except:
            pass

        del test_input, recons  # , samples

    def configure_optimizers(self):

        optims = []
        scheds = []

        optimizer_dict = {'adagrad': Adagrad, 'adam': Adam, 'sgd': SGD}

        assert self.params['optimizer'] in [
            *optimizer_dict], f'only support {[*optimizer_dict]} as optimizers'

        optimizer = optimizer_dict[self.params['optimizer']](
            self.model.parameters(),
            lr=self.params['LR'],
            weight_decay= float(self.params['weight_decay']),
            eps = float(self.params['eps']),
            betas=(0.9, 0.999)
        )

        optims.append(optimizer)
        # Check if more than 1 optimizer is required (Used for adversarial training)
        try:
            if self.params['LR_2'] is not None:
                optimizer2 = optim.Adam(getattr(self.model, self.params['submodel']).parameters(),
                                        lr=self.params['LR_2'])
                optims.append(optimizer2)
        except:
            pass

        try:
            if self.params['scheduler_gamma'] is not None:
                scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
                                                             gamma=self.params['scheduler_gamma'])
                scheds.append(scheduler)

                # Check if another scheduler is required for the second optimizer
                try:
                    if self.params['scheduler_gamma_2'] is not None:
                        scheduler2 = optim.lr_scheduler.ExponentialLR(optims[1],
                                                                      gamma=self.params['scheduler_gamma_2'])
                        scheds.append(scheduler2)
                except:
                    pass
                return optims, scheds
            else:
                # if no schedular gama, reduce LR by factor (0.1) when a metric has stopped improving
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optims[0], factor=0.1, patience=10, verbose=True)
                scheds.append(scheduler)
        except:
            return optims

    @data_loader
    def train_dataloader(self):

        if self.params['dataset'] == 'celeba':
            transform = self.data_transforms()
            dataset = CelebA(root=self.params['data_path'],
                             split="train",
                             transform=transform,
                             download=False)
            self.num_train_imgs = len(dataset)
            return DataLoader(dataset,
                              batch_size=self.params['batch_size'],
                              shuffle=True,
                              drop_last=True)

        elif 'codex' in self.params['dataset'] and 'num_cluster' in self.params:
            root = os.path.join(
                self.params['data_path'], self.params['dataset'])
            dataset = CodeX_Landmark_Dataset(root, self.data_transforms(), num_cluster = self.params['num_cluster'], item_number=self.params['cell_number'])
            self.num_train_imgs = len(dataset)
            return DataLoader(dataset,
                              batch_size=self.params['batch_size'],
                              shuffle=True,
                              drop_last=True)
        elif 'codex' in self.params['dataset']:
            root = os.path.join(
                self.params['data_path'], self.params['dataset'])
            dataset = CodeX_Dataset(root, self.data_transforms(), norm_method= self.params['norm_method'], in_channels=self.params['in_channels'], item_number=self.params['cell_number'])
            self.num_train_imgs = len(dataset)
            return DataLoader(dataset,
                              batch_size=self.params['batch_size'],
                              shuffle=True,
                              drop_last=True)

        elif self.is_IDC_dataset(self.params['dataset']):
            root = os.path.join(
                self.params['data_path'], self.params['dataset'])
            dataset = IDC_Dataset(root=root, transform=self.data_transforms())
            self.num_train_imgs = len(dataset)
            return DataLoader(dataset,
                              batch_size=self.params['batch_size'],
                              shuffle=True,
                              drop_last=True)

        elif self.is_hic_dataset(self.params['dataset']):
            root = os.path.join(
                self.params['data_path'], self.params['dataset'])
            dataset = HiC_Dataset(root=root, transform=self.data_transforms())
            self.num_train_imgs = len(dataset)

            if 'weighted_sampler' in self.params and self.params['weighted_sampler']:
                # resampling training data based on the inital size
                # upsampling by 3 if the size if larger than 20 * 10K
                weights = torch.tensor([1, 3])

                def get_weight(labels, sample_index, weights):
                    label = labels.iloc[sample_index]
                    return weights[1 if (label['end'] - label['start'] > 24) else 0]
                size_sampler = WeightedRandomSampler(
                    weights=[get_weight(dataset.img_labels, i, weights)
                             for i in range(self.num_train_imgs)],
                    num_samples=self.num_train_imgs,
                    replacement=True
                )

                return DataLoader(dataset,
                                  batch_size=self.params['batch_size'],
                                  # shuffle = True, # no shuffle if sampler is used
                                  sampler=size_sampler,
                                  drop_last=True)
            else:
                return DataLoader(dataset,
                                  batch_size=self.params['batch_size'],
                                  shuffle=True,
                                  drop_last=True)

        elif self.is_tensor_dataset(self.params['dataset']):
            print('start train data loading')
            root = os.path.join(
                self.params['data_path'], f"{self.params['dataset']}.npz")
            if not os.path.exists(root):
                import subprocess
                print('Now download dsprites-dataset')
                subprocess.call(['./download_dsprites.sh'])
                print('Finished')
            data = np.load(root, encoding='bytes')
            if 'dsprites' in self.params['dataset']:
                tensor = torch.from_numpy(data['imgs']).unsqueeze(1)
                labels = torch.from_numpy(data['latents_classes'])
            else:
                # unsequeeze reshape data from [x, 64, 64] to [x, 1, 64, 64]
                tensor = torch.from_numpy(data['imgs']).unsqueeze(1)
                labels = torch.from_numpy(data['labels'])

            # transform = self.data_transforms()
            # tensor = transform(tensor)

            train_kwargs = {'data_tensor': tensor, 'labels': labels}
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
            self.sample_dataloader = DataLoader(CelebA(root=self.params['data_path'],
                                                       split="test",
                                                       transform=transform,
                                                       download=False),
                                                batch_size=self.params['batch_size'],
                                                shuffle=True,
                                                drop_last=True)
            self.num_val_imgs = len(self.sample_dataloader)
            return self.sample_dataloader

        elif 'codex' in self.params['dataset'] and 'num_cluster' in self.params:
            root = os.path.join(
                self.params['data_path'], self.params['dataset'])
            dataset = CodeX_Landmark_Dataset(root, self.data_transforms(), num_cluster = self.params['num_cluster'], item_number=self.params['cell_number'])
            self.num_val_imgs = len(dataset)
            self.sample_dataloader = DataLoader(dataset,
                                                batch_size=self.params['batch_size'],
                                                shuffle=True,
                                                drop_last=True)
            return self.sample_dataloader

        elif 'codex' in self.params['dataset']:
            root = os.path.join(
                self.params['data_path'], self.params['dataset'])
            dataset = CodeX_Dataset(root, self.data_transforms(), norm_method= self.params['norm_method'], in_channels=self.params['in_channels'], item_number=self.params['cell_number'])
            self.num_val_imgs = len(dataset)
            self.sample_dataloader = DataLoader(dataset,
                                                batch_size=self.params['batch_size'],
                                                shuffle=True,
                                                drop_last=True)
            return self.sample_dataloader

        elif self.is_hic_dataset(self.params['dataset']) or self.is_tensor_dataset(self.params['dataset']) or self.is_IDC_dataset(self.params['dataset']):
            print('start val data loading')
            # since the two datasets are small, use the train data loader for val
            self.sample_dataloader = self.train_dataloader()
            self.num_val_imgs = self.num_train_imgs
            return self.sample_dataloader
        else:
            raise ValueError('Undefined dataset type')

    @data_loader
    def test_dataloader(self):
        print('loading test data')
        if self.params['dataset'] == 'celeba':
            transform = self.test_data_transforms()
            dataset = CelebA(root=self.params['data_path'],
                             split="train",
                             transform=transform,
                             download=False)
            dataset = Subset(dataset, list(range(2400)))
            self.test_sample_dataloader = DataLoader(dataset,
                                                     batch_size=self.params['batch_size'],
                                                     shuffle=False,
                                                     drop_last=True)
            self.num_test_imgs = len(self.test_sample_dataloader)
            return self.test_sample_dataloader

        
        elif 'codex' in self.params['dataset'] and 'num_cluster' in self.params:
            root = os.path.join(
                self.params['data_path'], self.params['dataset'])
            dataset = CodeX_Landmark_Dataset(root, self.data_transforms(), num_cluster = self.params['num_cluster'], item_number=self.params['cell_number'])
            self.num_test_imgs = len(dataset)
            return DataLoader(dataset,
                              batch_size=self.params['batch_size'],
                              shuffle=True,
                              drop_last=True)

        elif 'codex' in self.params['dataset']:
            root = os.path.join(
                self.params['data_path'], self.params['dataset'])
            dataset = CodeX_Dataset(root, self.data_transforms(), norm_method= self.params['norm_method'], in_channels=self.params['in_channels'], item_number=self.params['cell_number'])
            self.num_test_imgs = len(dataset)
            return DataLoader(dataset,
                              batch_size=self.params['batch_size'],
                              shuffle=True,
                              drop_last=True)

        elif self.is_hic_dataset(self.params['dataset']):

            root = os.path.join(
                self.params['data_path'], self.params['dataset'])
            dataset = HiC_Dataset(
                root=root, transform=self.test_data_transforms())
            self.test_sample_dataloader = DataLoader(dataset,
                                                     batch_size=self.params['batch_size'],
                                                     shuffle=False,
                                                     drop_last=False)

            self.num_test_imgs = len(self.test_sample_dataloader)
            return self.test_sample_dataloader

        elif self.is_IDC_dataset(self.params['dataset']):

            root = os.path.join(
                self.params['data_path'], self.params['dataset'])
            dataset = IDC_Dataset(
                root=root, transform=self.test_data_transforms())
            self.test_sample_dataloader = DataLoader(dataset,
                                                     batch_size=self.params['batch_size'],
                                                     shuffle=False,
                                                     drop_last=True)

            self.num_test_imgs = len(self.test_sample_dataloader)
            return self.test_sample_dataloader

        elif self.is_tensor_dataset(self.params['dataset']):
            root = os.path.join(
                self.params['data_path'], f"{self.params['dataset']}.npz")
            if not os.path.exists(root):
                import subprocess
                print('Now download dsprites-dataset')
                subprocess.call(['./download_dsprites.sh'])
                print('Finished')
            data = np.load(root, encoding='bytes')
            if 'dsprites' in self.params['dataset']:
                tensor = torch.from_numpy(data['imgs']).unsqueeze(1)
                labels = torch.from_numpy(data['latents_classes'])
            else:
                # unsequeeze reshape data from [x, 64, 64] to [x, 1, 64, 64]
                tensor = torch.from_numpy(data['imgs']).unsqueeze(1)
                labels = torch.from_numpy(data['labels'])

            # transform = self.data_transforms()
            # tensor = transform(tensor)

            train_kwargs = {'data_tensor': tensor, 'labels': labels}
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
        SetRange = transforms.Lambda(lambda X: 2 * X - 1.)  # [0,1] to [-1, 1]

        if self.is_hic_dataset(self.params['dataset']):
            transform = transforms.Compose([
                transforms.Resize(self.params['img_size'], Image.NEAREST),
                transforms.RandomApply([transforms.RandomHorizontalFlip(
                    1), transforms.RandomVerticalFlip(1)], 0.5),
                transforms.ToTensor(),
                SetRange])
        elif 'codex' in self.params['dataset']:
            vFlip = transforms.RandomApply(
                [transforms.Lambda(lambda X: np.flip(X, axis=1).copy())], 0.5)
            hFlip = transforms.RandomApply(
                [transforms.Lambda(lambda X: np.flip(X, axis=2).copy())], 0.5)
            np2Tensor = transforms.Lambda(lambda X: torch.tensor(X))
            transform = transforms.Compose([
                vFlip,
                hFlip,
                np2Tensor,
                SetRange
            ])
        elif self.is_IDC_dataset(self.params['dataset']):
            transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                            transforms.RandomVerticalFlip(),
                                            transforms.Resize(
                                                (self.params['img_size'], self.params['img_size'])),
                                            SetRange])

        elif self.params['dataset'] == 'celeba':
            transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                            transforms.CenterCrop(148),
                                            transforms.Resize(
                                                self.params['img_size']),
                                            transforms.ToTensor(),
                                            SetRange])

        else:
            # do not use transforms for tensor datasets for now
            transform = transforms.Compose([
                SetRange
            ])
        return transform

    def test_data_transforms(self):

        SetRange = transforms.Lambda(lambda X: 2 * X - 1.)  # [0,1] to [-1, 1]
        np2Tensor = transforms.Lambda(lambda X: torch.tensor(X))
        if self.params['dataset'] == 'celeba':

            transform = transforms.Compose([transforms.CenterCrop(148),
                                            transforms.Resize(
                                                self.params['img_size']),
                                            transforms.ToTensor(),
                                            SetRange])

        elif 'codex' in self.params['dataset']:
            transform = transforms.Compose([
                np2Tensor,
                SetRange
            ])

        elif self.is_hic_dataset(self.params['dataset']):
            transform = transforms.Compose([transforms.Resize(self.params['img_size'], Image.NEAREST),
                                            transforms.ToTensor(),
                                            SetRange])

        elif self.is_IDC_dataset(self.params['dataset']):
            transform = transforms.Compose([transforms.Resize((self.params['img_size'], self.params['img_size'])),
                                            transforms.ToTensor(),
                                            SetRange])

        else:
            #
            # do not use transforms for tensor datasets for now
            transform = transforms.Compose([
                SetRange
            ])
        return transform
