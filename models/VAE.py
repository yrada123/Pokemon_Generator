from typing import Any

from pytorch_lightning.utilities.types import STEP_OUTPUT, OptimizerLRScheduler

import scripts.config as conf
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

vae_yaml = conf.VAE_YAML

####################
#    Encoder       #
####################

class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, channels_dim):
        '''Setting encoder layers for VAE

        Args:
            input_dim (int): The spatial dimension of an input square image
            latent_dim (int): The latent vector dimension
            channels_dim (list): Number of output channels for each convolutional layer
        '''

        super().__init__()
        in_channels = 3
        encoder_conv_layers = []

        # Set conv layers
        for dim in channels_dim:
            encoder_conv_layers.append(nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=dim,
                          kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(dim),
                nn.LeakyReLU()
            ))
            in_channels = dim

        self.conv_layers = nn.Sequential(*encoder_conv_layers)

        # Set FC layers for (mu, logvar) -> (expectation, log variance)
        final_h_w = input_dim >> len(channels_dim)
        linear_dim = channels_dim[-1] * final_h_w ** 2
        self.fc_mu = nn.Linear(linear_dim, latent_dim)
        self.fc_logvar = nn.Linear(linear_dim, latent_dim)

    def forward(self, x):
        '''Feed forward the encoder

        Args:
            x (Tensor): encoder input image
        Return:
             z (Tensor): resulted latent
             mu (Tensor): µ - expectations vector
             std (Tensor): σ - standard deviation vector
        '''
        # conv layers -> expectation fc
        #                log var variance fc -> Gaussian sample latent
        conv_output = self.conv_layers(x)
        conv_output = nn.Flatten(conv_output)
        mu = self.fc_mu(conv_output)
        std = torch.exp(0.5 * self.fc_logvar(conv_output))
        eps = torch.randn_like(std)
        z = mu + std * eps
        return [z, mu, std]

####################
#    Decoder       #
####################

class Decoder(nn.Module):
    def __init__(self, input_dim, latent_dim, channels_dim):
        '''Setting decoder layers for VAE

            Args:
                input_dim (int): The spatial dimension of the encoder input square image
                latent_dim (int): The latent vector dimension
                channels_dim (list): Number of output channels for each convolutional layer
            '''
        super().__init__()

        # Set FC to increase dimensions
        self.in_channels = channels_dim[0]
        self.initial_h_w = input_dim >> len(channels_dim)
        linear_dim = self.in_channels * self.initial_h_w ** 2
        self.fc_input = nn.Linear(latent_dim, linear_dim)

        # Set conv layers
        in_channels = self.in_channels
        decoder_conv_layers = []

        for dim in channels_dim:
            decoder_conv_layers.append(nn.Sequential(
                nn.ConvTranspose2d(in_channels=in_channels, out_channels=dim,
                                   kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(dim),
                nn.LeakyReLU()
            ))
            in_channels = dim

        # Set the final layers to create the image
        final_layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=3,
                      kernel_size=3, padding='same'),
            nn.Tanh()
        )
        self.conv_layers = nn.Sequential(*decoder_conv_layers, final_layer)

    def forward(self, z):
        '''Feed forward the Decoder

        Args:
            z (tensor): encoded latent
        Return:
            (tensor): reconstructed image
        '''
        flat_initial = self.fc_input(z)
        return self.conv_layers(flat_initial.view(-1, self.in_channels, self.initial_h_w, self.initial_h_w))

###############
#   VAE       #
###############

class VAE(pl.LightningModule):
    def __init__(self):
        '''Set Complete VAE'''
        super().__init__()

        ### Set model's values ###
        self.latent_dim = vae_yaml['model_params']['latent_dim']
        self.input_dim = vae_yaml['data_params']['img_size']
        self.channels_dim = vae_yaml['model_params']['output_channels_dim_default']
        self.w_kld = vae_yaml['training_params']['w_kld']

        self.encoder = Encoder(self.input_dim, self.latent_dim, self.channels_dim)
        self.decoder = Decoder(self.input_dim, self.latent_dim, self.channels_dim)

    # Interface

    def forward(self, x):
        '''Feed forward the VAE

        Args:
            x (tensor): VAE input image
        Return:
            rec_x (tensor): reconstructed image
            x (tensor): original input image
            mu (tensor): expectation of latent
            std (tensor): standard variation of latent
        '''
        z, mu, std = self.encoder(x)
        rec_x = self.decoder(z)
        return [rec_x, x, mu, std]

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    # Train

    def training_step(self, batch, batch_idx):
        x = batch
        rec_x, _, mu, std = self(x)
        mse_loss = F.mse_loss(x, rec_x)
        kld_loss = torch.mean(-0.5 * torch.sum(mu**2 + std**2 -1 - 2*std.log(), dim=1), dim=0)
        loss = mse_loss + self.w_kld * kld_loss
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        rec_x, _, mu, std = self(x)
        mse_loss = F.mse_loss(x, rec_x)
        kld_loss = torch.mean(-0.5 * torch.sum(mu ** 2 + std ** 2 - 1 - 2 * std.log(), dim=1), dim=0)
        loss = mse_loss + self.w_kld * kld_loss
        return loss
    def configure_optimizers(self):
        lr = vae_yaml['training_params']["learning_rate"]
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        return optimizer










