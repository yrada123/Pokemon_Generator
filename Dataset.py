from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
import os
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
import pytorch_lightning as pl


class ImageDataset(Dataset):
    def __init__(self, root, transform=None):
        self.data_dir = root
        self.img_list = [x for x in os.listdir(root) if x.endswith('.png')]
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        name = self.img_list[idx]
        img = Image.open(os.path.join(self.data_dir, name)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img


class DataModule(pl.LightningDataModule):
    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        self.data_dir = self.conf['data_params']['data_dir']
        self.batch_size = self.conf['data_params']['batch_size']

    def setup(self, stage: str) -> None:
        img_size = self.conf['data_params']['img_size']
        transform = transforms.Compose([transforms.Resize((img_size, img_size)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset = ImageDataset(root=self.data_dir, transform=transform)

        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(dataset, [train_size, val_size])

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(dataset=self.val_dataset, batch_size=self.batch_size)
