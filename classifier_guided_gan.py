import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import matplotlib.pyplot as plt

class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim + 10, 128),  # Latent vector + one-hot label
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, int(torch.prod(torch.tensor(img_shape)))),
            nn.Tanh()
        )
        self.img_shape = img_shape

    def forward(self, z, labels):
        z = torch.cat((z, labels), dim=1)  # Concatenate latent vector and one-hot labels
        img = self.model(z)
        return img.view(img.size(0), *self.img_shape)

class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(int(torch.prod(torch.tensor(img_shape))) + 10, 512),  # Image flattened + one-hot label
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        img_flat = torch.flatten(img, start_dim=1)
        input_combined = torch.cat((img_flat, labels), dim=1)
        return self.model(input_combined)


class ClassifierGuidedGAN(pl.LightningModule):
    def __init__(self, latent_dim, img_shape, classifier):
        super().__init__()
        self.generator = Generator(latent_dim, img_shape)
        self.discriminator = Discriminator(img_shape)
        self.classifier = classifier
        self.latent_dim = latent_dim
        self.img_shape = img_shape
        self.automatic_optimization = False

        self.adversarial_loss = nn.BCELoss()
        self.classification_loss = nn.CrossEntropyLoss()

    def forward(self, z, labels):
        return self.generator(z, labels)

    def generate_digit(self, label):
        one_hot_label = torch.zeros(1, 10, device=self.device)
        one_hot_label[0, label] = 1
        z = torch.randn(1, self.latent_dim, device=self.device)
        generated_img = self.generator(z, one_hot_label)
        img = generated_img.squeeze().detach().cpu().numpy()
        plt.imshow(img, cmap="gray")
        plt.title(f"Generated Digit: {label}")
        plt.axis("off")
        plt.show()

    def adversarial_step(self, imgs, labels, valid, fake):
        # Train Discriminator
        real_loss = self.adversarial_loss(self.discriminator(imgs, labels), valid)
        z = torch.randn(imgs.size(0), self.latent_dim, device=self.device)
        fake_imgs = self.generator(z, labels)
        fake_loss = self.adversarial_loss(self.discriminator(fake_imgs.detach(), labels), fake)
        d_loss = (real_loss + fake_loss) / 2

        # Train Generator
        g_loss_adv = self.adversarial_loss(self.discriminator(fake_imgs, labels), valid)

        # Classification Loss
        class_preds = self.classifier(fake_imgs)
        g_loss_class = self.classification_loss(class_preds, labels.argmax(dim=1))

        # Schedule the weighting of the two losses over epochs
        current_epoch = self.current_epoch if self.trainer is not None else 0
        max_epochs = self.trainer.max_epochs if self.trainer is not None else 100
        progress = min(float(current_epoch) / float(max_epochs - 1), 1.0)

        # Define the weighting schedule
        weight_adv = 0.9 - 0.4 * progress  # goes from 0.9 at start to 0.5 at end
        weight_class = 0.1 + 0.4 * progress  # goes from 0.1 at start to 0.5 at end

        g_loss = weight_adv * g_loss_adv + weight_class * g_loss_class
        return d_loss, g_loss

    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        one_hot_labels = torch.zeros(labels.size(0), 10, device=self.device)
        one_hot_labels[torch.arange(labels.size(0)), labels] = 1
        valid = torch.ones(imgs.size(0), 1, device=self.device)
        fake = torch.zeros(imgs.size(0), 1, device=self.device)

        # Access optimizers manually:
        opt_d, opt_g = self.optimizers()

        opt_d.zero_grad()  # Zero discriminator gradients
        d_loss, _ = self.adversarial_step(imgs, one_hot_labels, valid, fake)
        self.manual_backward(d_loss)  # Manually backpropagate discriminator loss
        opt_d.step()  # Update discriminator parameters
        self.log("d_loss", d_loss, prog_bar=True)

        opt_g.zero_grad()  # Zero generator gradients
        _, g_loss = self.adversarial_step(imgs, one_hot_labels, valid, fake)
        self.manual_backward(g_loss)  # Manually backpropagate generator loss
        opt_g.step()  # Update generator parameters
        self.log("g_loss", g_loss, prog_bar=True)

    def configure_optimizers(self):
        optimizer_g = optim.Adam(self.generator.parameters(), lr=1e-4)
        optimizer_d = optim.Adam(self.discriminator.parameters(), lr=1e-4)
        return [optimizer_d, optimizer_g], []
