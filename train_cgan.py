import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import pytorch_lightning as pl

from classifier_guided_gan import ClassifierGuidedGAN
from cnn_classifier import CNNClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

latent_dim = 100
img_shape = (1, 28, 28)
batch_size = 64

# Data Preparation
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
dataset = datasets.MNIST(root="data", train=True, download=True, transform=transform)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Train Classifier
# classifier = CNNClassifier().to(device)
# trainer = pl.Trainer(max_epochs=5, accelerator="gpu", devices=1)
# trainer.fit(classifier, train_loader)

# Load Pretrained Classifier
classifier = CNNClassifier()
classifier.load_state_dict(torch.load('models/cnn_model_params.pth'))

# Freeze Classifier
for param in classifier.parameters():
    param.requires_grad = False

# Train GAN
gan = ClassifierGuidedGAN(latent_dim, img_shape, classifier)
trainer = pl.Trainer(max_epochs=50, accelerator="gpu", devices=1)
trainer.fit(gan, train_loader)

# Generate and Plot a Digit
gan.generate_digit(label=5)


# TODO - copy and paste from colab notebook