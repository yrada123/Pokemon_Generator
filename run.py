from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from models.VAE import VAE
from Dataset import DataModule
from pytorch_lightning.loggers import TensorBoardLogger

vae_model = VAE()
dm = DataModule()
logger = TensorBoardLogger("tb_logs", name="vae_model")

trainer = Trainer(logger=logger,
                  log_every_n_steps=1,
                  callbacks=[LearningRateMonitor(),
                             ModelCheckpoint(save_top_k=2,
                                             dirpath='outputs',
                                             monitor="kld_loss",
                                             save_last=True),
                             ],
                  accelerator='gpu',
                  devices=1,
                  max_epochs=15000)

trainer.fit(vae_model, dm, ckpt_path='outputs/last.ckpt')
