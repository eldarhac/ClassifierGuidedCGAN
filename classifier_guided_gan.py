import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from pathlib import Path
import torchvision.utils as vutils
from torchmetrics.classification import Accuracy


class Generator(nn.Module):
    """
    Conditional generator that upsamples latent vectors to images of arbitrary shape
    """
    def __init__(self, latent_dim: int, img_shape: tuple, num_classes: int = 10):
        super().__init__()
        self.latent_dim = latent_dim
        self.img_shape = img_shape
        self.num_classes = num_classes
        self.label_emb = nn.Embedding(num_classes, num_classes)

        _, h, w = img_shape
        self.init_size = min(h, w) // 4
        self.l1 = nn.Linear(latent_dim + num_classes, 128 * self.init_size**2)
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, img_shape[0], 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, z, labels):
        label_embedding = self.label_emb(labels)
        gen_input = torch.cat((z, label_embedding), dim=1)
        out = self.l1(gen_input)
        out = out.view(out.size(0), 128, self.init_size, self.init_size)
        return self.conv_blocks(out)


class Discriminator(nn.Module):
    """
    Conditional discriminator that downsamples images and concatenates label embeddings
    """
    def __init__(self, img_shape: tuple, num_classes: int = 10):
        super().__init__()
        self.img_shape = img_shape
        self.num_classes = num_classes
        self.label_emb = nn.Embedding(num_classes, num_classes)

        in_c = img_shape[0]
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(in_c, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.adv_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 + num_classes, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        features = self.conv_blocks(img)
        flat = features.view(features.size(0), -1)
        label_embedding = self.label_emb(labels)
        return self.adv_layer(torch.cat((flat, label_embedding), dim=1))


class ClassifierGuidedGAN(pl.LightningModule):
    def __init__(self,
                 latent_dim: int,
                 img_shape: tuple,
                 classifier: nn.Module,
                 num_classes: int = 10,
                 lr: float = 2e-4,
                 b1: float = 0.5,
                 b2: float = 0.999,
                 lambda_class: float = 0.5):
        super().__init__()
        self.save_hyperparameters(ignore=['classifier'])
        self.latent_dim = latent_dim
        self.img_shape = img_shape
        self.num_classes = num_classes
        self.lr = lr
        self.b1, self.b2 = b1, b2
        self.lambda_class = lambda_class

        self.generator = Generator(latent_dim, img_shape, num_classes)
        self.discriminator = Discriminator(img_shape, num_classes)

        # frozen classifier for inferring real labels
        self.classifier = classifier
        self.classifier.freeze()
        # for p in self.classifier.parameters(): p.requires_grad = False

        self.automatic_optimization = False
        self.adversarial_loss = nn.BCELoss()
        self.classification_loss = nn.CrossEntropyLoss()
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)

        self.fixed_z = torch.randn(num_classes, latent_dim)
        self.fixed_labels = torch.arange(0, num_classes, dtype=torch.long)

    def forward(self, z, labels):
        return self.generator(z, labels)

    def configure_optimizers(self):
        opt_g = optim.Adam(self.generator.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        opt_d = optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        return [opt_g, opt_d]

    def training_step(self, batch, batch_idx):
        imgs, _ = batch  # ignore dataset labels
        batch_size = imgs.size(0)

        # infer labels via frozen classifier
        with torch.no_grad():
            inferred_labels = self.classifier(imgs).argmax(dim=1)

        opt_g, opt_d = self.optimizers()
        valid = torch.ones(batch_size, 1, device=self.device)
        fake = torch.zeros(batch_size, 1, device=self.device)

        # Train Discriminator
        self.toggle_optimizer(opt_d)
        z = torch.randn(batch_size, self.latent_dim, device=self.device)
        gen_labels = torch.randint(0, self.num_classes, (batch_size,), device=self.device)
        fake_imgs = self.generator(z, gen_labels)

        real_pred = self.discriminator(imgs, inferred_labels)
        fake_pred = self.discriminator(fake_imgs.detach(), gen_labels)
        d_loss = (self.adversarial_loss(real_pred, valid) + self.adversarial_loss(fake_pred, fake)) / 2

        opt_d.zero_grad()
        self.manual_backward(d_loss)
        opt_d.step()
        self.untoggle_optimizer(opt_d)
        self.log('d_loss', d_loss, prog_bar=True)

        # Train Generator
        self.toggle_optimizer(opt_g)
        z = torch.randn(batch_size, self.latent_dim, device=self.device)
        gen_labels = torch.randint(0, self.num_classes, (batch_size,), device=self.device)
        fake_imgs = self.generator(z, gen_labels) # Need gradients now

        fake_pred = self.discriminator(fake_imgs, gen_labels)
        g_loss_adv = self.adversarial_loss(fake_pred, valid) # Try to fool discriminator

        # Classification Loss (Classifier's prediction on fake images)
        class_preds = self.classifier(fake_imgs)
        g_loss_class = self.classification_loss(class_preds, gen_labels) # Use generated labels

        # Classification Loss (Classifier's prediction on fake images)
        class_preds = self.classifier(fake_imgs)
        g_loss_class = self.classification_loss(class_preds, gen_labels) # Use generated labels

        # Total generator loss
        # Using fixed lambda_class, but could implement schedule like before if needed
        g_loss = g_loss_adv + self.lambda_class * g_loss_class


        opt_g.zero_grad()
        self.manual_backward(g_loss)
        opt_g.step()
        self.untoggle_optimizer(opt_g)
        self.log('g_loss', g_loss, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        z = torch.randn(self.num_classes, self.latent_dim, device=self.device)
        labels = torch.arange(self.num_classes, device=self.device)
        imgs = self.generator(z, labels)
        preds = self.classifier(imgs)
        self.val_acc.update(preds, labels)

    def on_validation_epoch_end(self):
        acc = self.val_acc.compute()
        self.log('val_acc', acc, prog_bar=True)
        self.val_acc.reset()

    def generate_images(self, epoch: int, output_dir: str = 'generated_images'):
        Path(output_dir).mkdir(exist_ok=True)
        with torch.no_grad():
            imgs = self.generator(self.fixed_z.to(self.device), self.fixed_labels.to(self.device))
        grid = vutils.make_grid(imgs, nrow=self.num_classes, normalize=True)
        vutils.save_image(grid, Path(output_dir) / f'epoch_{epoch}.png')


