from configs.config import load_yaml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from models.VAE import VAE
from Dataset import DataModule
from pytorch_lightning.loggers import TensorBoardLogger
import os
import shutil
import string
import random

conf = load_yaml('VAE.yaml')

w_klds = [10e-3]
latents = [128]
convs = [3]

for kld in w_klds:
    for latent in latents:
        for conv in convs:

            vae_model = VAE(conf)
            dm = DataModule(conf)
            logger = TensorBoardLogger(conf["logging_params"]["log_dir"], name="vae_model")

            vae_model.w_kld = kld
            vae_model.latent_dim = latent
            for i in range(len(vae_model.conv_channels)):
                vae_model.conv_channels[i][1] = conv

            trainer = Trainer(logger=logger,
                              log_every_n_steps=1,
                              callbacks=[ModelCheckpoint(save_top_k=2,
                                                         dirpath='outputs',
                                                         monitor="val_loss",
                                                         save_last=True),
                                         ],
                              check_val_every_n_epoch=conf["trainer_params"]["check_val_every_n_epoch"],
                              accelerator='gpu',
                              devices=1,
                              max_epochs=conf["trainer_params"]["max_epochs"])
            trainer.fit(vae_model, dm, ckpt_path=conf["trainer_params"]["ckpt"])


            dir_name = 'kld=' + str(kld) + ', latent=' + str(latent) + ', conv=' + str(conv)+ ', do=' + str(conf['model_params']['dropout'])
            dest = os.path.join('outputs', dir_name)
            files = [i for i in os.listdir('outputs') if i.endswith('ckpt')]
            try:
                os.mkdir(dest)
            except:
                dir_name += ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(3))
                dest = dest + dir_name
                os.mkdir(dest)
            for file in files:
                shutil.move(os.path.join('outputs', file), dest)
            os.rename('tb_logs/vae_model/version_0', 'tb_logs/vae_model/' + dir_name)




