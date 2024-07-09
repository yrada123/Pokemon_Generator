from configs.config import load_yaml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from models.VAE import VAE
from Dataset import DataModule
from pytorch_lightning.loggers import TensorBoardLogger

conf = load_yaml('VAE.yaml')

vae_model = VAE(conf)
dm = DataModule(conf)
logger = TensorBoardLogger(conf["logging_params"]["log_dir"], name="vae_model")

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
