import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl


class CNNClassifier(pl.LightningModule):
    """
    A general CNN classifier that adapts to different input channels and image sizes
    using adaptive pooling. Can be used for both MNIST (1-channel) and CIFAR10 (3-channel).
    """
    def __init__(self, in_channels: int = 1, num_classes: int = 10, lr: float = 1e-3):
        super().__init__()
        # save hyperparameters for easy access
        self.save_hyperparameters()
        self.lr = lr
        # Feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

        )
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )
        # Loss function
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.features(x)  # [B,C,H,W] on MPS
        if x.device.type == "mps":
            # pull to CPU, do the pooling, push back
            x = nn.AdaptiveAvgPool2d((4, 4))(x.cpu()).to("mps")
        else:
            x = nn.AdaptiveAvgPool2d((4, 4))(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (torch.argmax(logits, dim=1) == y).float().mean()
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log('val_loss', loss, on_step=False, prog_bar=True, on_epoch=True, logger=True)
        self.log('val_acc', acc, on_step=False, prog_bar=True, on_epoch=True, logger=True)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)


class HistoryCallback(pl.Callback):
    def on_train_start(self, trainer, pl_module):
        pl_module.train_losses = []
        pl_module.train_accs   = []
        pl_module.val_losses   = []
        pl_module.val_accs     = []

    def on_train_epoch_end(self, trainer, pl_module):
        m = trainer.callback_metrics
        pl_module.train_losses.append(
            m.get('train_loss_epoch', m.get('train_loss')).item()
        )
        pl_module.train_accs.append(
            m.get('train_acc_epoch', m.get('train_acc')).item()
        )

    def on_validation_epoch_end(self, trainer, pl_module):
        m = trainer.callback_metrics
        # use the `_epoch` version if present, otherwise fall back
        loss = m.get('val_loss_epoch', m.get('val_loss'))
        acc  = m.get('val_acc_epoch',  m.get('val_acc'))
        pl_module.val_losses.append(loss.item())
        pl_module.val_accs.append( acc.item())
