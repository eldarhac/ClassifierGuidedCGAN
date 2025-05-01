import pytorch_lightning as pl
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class ClassificationDataModule(pl.LightningDataModule):
    """
    DataModule that supports MNIST and CIFAR10 based on dataset_name.
    Automatically sets in_channels for the model.
    """
    def __init__(self, dataset_name: str, data_dir: str = './data', batch_size: int = 64):
        super().__init__()
        self.dataset_name = dataset_name.lower()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, stage=None):
        if self.dataset_name == 'mnist':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            self.train_dataset = datasets.MNIST(self.data_dir, train=True, download=True, transform=transform)
            self.val_dataset = datasets.MNIST(self.data_dir, train=False, download=True, transform=transform)
            self.in_channels = 1
        elif self.dataset_name == 'cifar10':
            # 1) Stronger train‐time augmentation
            train_tf = transforms.Compose([
                transforms.RandomCrop(32, padding=4),          # random 32×32 crop from padded 40×40
                transforms.RandomHorizontalFlip(),             # random left↔right flip
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465),
                    (0.2023, 0.1994, 0.2010)
                )
            ])
            # 2) No augmentation on validation
            val_tf = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465),
                    (0.2023, 0.1994, 0.2010)
                )
            ])

            self.train_dataset = datasets.CIFAR10(self.data_dir, train=True,
                                                  download=True, transform=train_tf)
            self.val_dataset   = datasets.CIFAR10(self.data_dir, train=False,
                                                  download=True, transform=val_tf)
            self.in_channels   = 3
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=7,
                          persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=7,
                          persistent_workers=True)

class GANDataModule(pl.LightningDataModule):
    """
    DataModule supporting MNIST and CIFAR10, providing real images and labels
    """
    def __init__(self, dataset_name: str, data_dir: str = './data', batch_size: int = 64):
        super().__init__()
        self.dataset_name = dataset_name.lower()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, stage=None):
        if self.dataset_name == 'mnist':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
            ds = datasets.MNIST(self.data_dir, train=True, download=True, transform=transform)
            val_ds = datasets.MNIST(self.data_dir, train=False, download=True, transform=transform)
            in_channels, size = 1, 28
        elif self.dataset_name == 'cifar10':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            ds = datasets.CIFAR10(self.data_dir, train=True, download=True, transform=transform)
            val_ds = datasets.CIFAR10(self.data_dir, train=False, download=True, transform=transform)
            in_channels, size = 3, 32
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")
        self.train_dataset = ds
        self.val_dataset = val_ds
        self.img_shape = (in_channels, size, size)
        self.num_classes = 10

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=7,
                          persistent_workers=True)
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=7,
                          persistent_workers=True)