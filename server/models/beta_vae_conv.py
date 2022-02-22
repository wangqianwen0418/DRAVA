import torch
from models import BaseVAE
from torch import nn
from torch.nn import functional as F
from .types_ import *

import math
import numpy as np


class BetaVAE_CONV(BaseVAE):

    num_iter = 0 # Global static variable to keep track of iterations

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 beta: int = 4,
                 gamma:float = 1000.,
                 loss_type:str = 'B',
                 img_size:int = 64,
                 max_capacity: int = 25, # works similar to the beta in original beta vae
                 Capacity_max_iter: int = 1e5,
                 **kwargs) -> None:
        super(BetaVAE_CONV, self).__init__()

        self.latent_dim = latent_dim
        self.beta = beta
        self.gamma = gamma
        self.loss_type = loss_type
        
        conv_sizes =  kwargs.get('conv_sizes', [ 3 for i in hidden_dims])
        self.recons_multi = kwargs.get('recons_multi', 1)
        is_masked = kwargs.get('is_masked', False)
        
        self.C_max = torch.Tensor([max_capacity])
        self.C_stop_iter = Capacity_max_iter
        out_channels = in_channels

        modules = []

        # for matrix images, we use a mask otherwise the diagonal is emphasized too much
        self.mask = np.ones( (img_size, img_size) )
        ratio = 0.5
        if is_masked:
            for i in range(img_size):
                for j in range(img_size):
                    self.mask[i,j] = ratio * abs(i-j)/63 + (1-ratio)
        self.mask = torch.from_numpy(self.mask).float()
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
        else:
            device = torch.device("cpu")
        self.mask = self.mask.to(device)


        # Build Encoder
        w_dim = img_size
        padding = 1
        stride = 2
        dilation = 1 
        for i, h_dim in enumerate(hidden_dims):
            kernel_size = conv_sizes[i]
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, h_dim,
                              kernel_size, stride, padding, dilation = dilation),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim
            # calculated as https://pytorch.org/docs/master/generated/torch.nn.Conv2d.html#torch.nn.Conv2d
            w_dim = math.floor((w_dim + 2 * padding - dilation * (kernel_size - 1) - 1)/stride + 1)
        
        self.encoder_outsize = [h_dim, w_dim, w_dim] # w and h is the same in our case
        flat_outsize = h_dim * w_dim * w_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear( flat_outsize, self.latent_dim )
        self.fc_var = nn.Linear( flat_outsize, self.latent_dim )


        # Build Decoder
        modules = []

        hidden_dims.reverse()
        conv_sizes.reverse()

        for i in range(len(hidden_dims) - 1):
            kernel_size= conv_sizes[i]
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )


        self.decoder_input = nn.Linear(latent_dim, flat_outsize) 
        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               out_channels,
                                               kernel_size=conv_sizes[-1],
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(out_channels),
                            nn.ReLU())

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        encoder_out = self.encoder(input)

        flat_out = torch.flatten(encoder_out, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        
        mu = self.fc_mu(flat_out)
        log_var = self.fc_var(flat_out)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:

        result = self.decoder_input(z)
        result = result.view(-1, *self.encoder_outsize) 
        result = self.decoder(result)
        result = self.final_layer(result)
        
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Will a single z be enough to compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        self.num_iter += 1
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset

        recons_loss =F.mse_loss(recons * self.mask, input * self.mask) * self.recons_multi

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        if self.loss_type == 'H': # https://openreview.net/forum?id=Sy2fzU9gl

            weighted_kld_loss = self.beta * kld_weight * kld_loss
            loss = recons_loss + weighted_kld_loss

        elif self.loss_type == 'B': # https://arxiv.org/pdf/1804.03599.pdf
            self.C_max = self.C_max.to(input.device)
            C = torch.clamp(self.C_max/self.C_stop_iter * self.num_iter, 0, self.C_max.data[0])
            weighted_kld_loss = self.gamma * kld_weight* (kld_loss - C).abs()
            loss = recons_loss + weighted_kld_loss
        else:
            raise ValueError('Undefined loss type.')

        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':kld_loss, 'weighted_KLD': weighted_kld_loss}

    def recons_loss(self,
                      *args,
                      **kwargs) -> dict:
        recons = args[0]
        input = args[1]

        recons_loss = F.mse_loss(recons, input, reduction='none') # reduction ='none' will return the mse loss for each sample
        recons_loss = recons_loss.view(recons_loss.size(0), -1).mean(1) # average except along dim 0
        return recons_loss

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples


    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]