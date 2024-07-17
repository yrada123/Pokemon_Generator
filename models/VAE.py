import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision.utils import make_grid

####################
#    Encoder       #
####################


class Encoder(nn.Module):
    def __init__(self, img_size, latent_dim, conv_channels, dropout=0.1):
        """Setting encoder layers for VAE

            Args:
                img_size (int): The spatial dimension of an input square image
                latent_dim (int): The latent vector dimension
                conv_channels (list): Number of output channels for each convolutional layer
        """

        super().__init__()
        in_channels = 3
        encoder_conv_layers = []

        # Set conv layers
        for ch, rep in conv_channels:
            for i in range(rep):
                is_last = i == rep-1
                stride = (1, 1) if not is_last else (2, 2)  # last layer halves spatial dim
                encoder_conv_layers.append(nn.Sequential(
                    nn.Conv2d(in_channels=in_channels, out_channels=ch,
                              kernel_size=(3, 3), stride=stride, padding=1),
                    nn.BatchNorm2d(ch),
                    nn.LeakyReLU(),
                    nn.Dropout(p=dropout) if is_last else nn.Identity()
                ))
                in_channels = ch
        self.conv_layers = nn.Sequential(*encoder_conv_layers)

        # Set FC layers for (mu, logvar) -> (expectation, log variance)
        final_h_w = img_size >> len(conv_channels)
        linear_dim = conv_channels[-1][0] * final_h_w ** 2
        self.fc_mu = nn.Linear(linear_dim, latent_dim)
        self.fc_logvar = nn.Linear(linear_dim, latent_dim)

    def forward(self, x):
        """Feed forward the encoder

            Args:
                x (Tensor): encoder input image
            Return:
                 z (Tensor): resulted latent
                 mu (Tensor): µ - expectations vector
                 logvar (Tensor): 2log(σ) - log variance vector
        """
        # conv layers -> expectation fc
        #                log var variance fc -> Gaussian sample latent
        conv_output = self.conv_layers(x)
        conv_output = torch.flatten(conv_output, start_dim=1)
        mu = self.fc_mu(conv_output)
        logvar = self.fc_logvar(conv_output)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps
        return [z, mu, logvar]

####################
#    Decoder       #
####################


class Decoder(nn.Module):
    def __init__(self, img_size, latent_dim, conv_channels, dropout=0.1):
        """Setting decoder layers for VAE

            Args:
                img_size (int): The spatial dimension of the encoder input square image
                latent_dim (int): The latent vector dimension
                conv_channels (list): Number of output channels for each convolutional layer
        """
        super().__init__()

        # Set FC to increase dimensions
        self.in_channels = conv_channels[0][0]
        self.initial_h_w = img_size >> len(conv_channels)
        linear_dim = self.in_channels * self.initial_h_w ** 2
        self.fc_input = nn.Linear(latent_dim, linear_dim)

        # Set conv layers
        in_channels = self.in_channels
        decoder_conv_layers = []

        for ch, rep in conv_channels:
            for i in range(rep):
                is_last = i == rep - 1
                if not is_last:
                    conv = nn.Conv2d(in_channels=in_channels, out_channels=ch,
                                     kernel_size=(3, 3), stride=(1, 1), padding=1)
                else:
                    conv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=ch,
                                              kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))
                decoder_conv_layers.append(nn.Sequential(
                    conv,
                    nn.BatchNorm2d(ch),
                    nn.LeakyReLU(),
                    nn.Dropout(p=dropout) if is_last else nn.Identity()
                ))
                in_channels = ch

        # Set the final layers to create the image
        final_layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=3,
                      kernel_size=(3, 3), padding='same'),
            nn.Tanh()
        )
        self.conv_layers = nn.Sequential(*decoder_conv_layers, final_layer)

    def forward(self, z):
        """Feed forward the Decoder

            Args:
                z (tensor): encoded latent
            Return:
                (tensor): reconstructed image
        """
        flat_initial = self.fc_input(z)
        return self.conv_layers(flat_initial.view(-1, self.in_channels, self.initial_h_w, self.initial_h_w))

###############
#   VAE       #
###############


class VAE(pl.LightningModule):
    def __init__(self, conf):
        """Set Complete VAE"""
        super().__init__()

        ### Set model's values ###
        self.conf = conf
        self.conv_channels = conf['model_params']['conv_channels']
        self.latent_dim = conf['model_params']['latent_dim']
        self.img_size = conf['data_params']['img_size']
        self.w_kld = conf['training_params']['w_kld']
        self.lr = conf['training_params']['learning_rate']
        self.dropout = conf["model_params"]["dropout"]

        self.encoder = Encoder(self.img_size, self.latent_dim, self.conv_channels, self.dropout)
        self.decoder = Decoder(self.img_size, self.latent_dim, self.conv_channels, self.dropout)

    # Interface

    def forward(self, x):
        """Feed forward the VAE

        Args:
            x (tensor): VAE input image
        Return:
            rec_x (tensor): reconstructed image
            x (tensor): original input image
            mu (tensor): expectation of latent
            logvar (tensor): log variance of latent
        """
        z, mu, logvar = self.encoder(x)
        rec_x = self.decoder(z)
        return [rec_x, x, mu, logvar]

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    # Train

    def training_step(self, batch, batch_idx):
        loss, mse_loss, kld_loss, rec_x, mu, logvar = self.loss_calc(batch)
        concatenated_images = torch.cat((batch[:4], rec_x[:4]), dim=3)
        grid = make_grid(concatenated_images)
        self.logger.experiment.add_image('train_images', grid, self.global_step)
        if self.current_epoch > 10:
            self.logger.experiment.add_histogram('train_mu', mu, self.current_epoch)
            self.logger.experiment.add_histogram('train_std', (logvar / 2).exp(), self.current_epoch)
        self.log('train_mse_loss', mse_loss)
        self.log('train_kld_loss', kld_loss)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, mse_loss, kld_loss, rec_x, mu, logvar = self.loss_calc(batch)
        concatenated_images = torch.cat((batch[:4], rec_x[:4]), dim=3)
        grid = make_grid(concatenated_images)
        self.logger.experiment.add_image('val_images', grid, self.global_step)
        if self.current_epoch > 10:
            self.logger.experiment.add_histogram('val_mu', mu, self.current_epoch)
            self.logger.experiment.add_histogram('val_std', (logvar / 2).exp(), self.current_epoch)
        self.log('val_mse_loss', mse_loss)
        self.log('val_kld_loss', kld_loss)
        self.log('val_loss', loss)
        return loss

    def loss_calc(self, batch):
        x = batch
        rec_x, _, mu, logvar = self(x)
        mse_loss = F.mse_loss(x, rec_x)
        kld_loss = torch.mean(0.5 * torch.mean(mu ** 2 + logvar.exp() - 1 - logvar, dim=1), dim=0)
        loss = mse_loss + self.w_kld * kld_loss
        return loss, mse_loss, kld_loss, rec_x, mu, logvar

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
