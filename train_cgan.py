import os
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import pytorch_lightning as pl
import torchvision.utils as vutils
import matplotlib.pyplot as plt

from classifier_guided_gan import ClassifierGuidedGAN
from cnn_classifier import CNNClassifier
from data_module import GANDataModule


def train_cnn_cgan(dataset_name, batch_size=64, max_epochs=10, lr=2e-4, latent_dim=100, clf_path=None, save_path=None):
    # 1. Data
    dm = GANDataModule(dataset_name=dataset_name, batch_size=batch_size)
    dm.setup()

    # 2. Load pre-trained classifier
    try:
        if clf_path is None:
            clf_path = f"models/{dataset_name}_classifier_params.pth"
        clf = CNNClassifier(in_channels=dm.in_channels, lr=lr)
        clf.load_state_dict(torch.load(clf_path))
    except FileNotFoundError:
        print(f"Classifier model not found at {clf_path}. Please train the classifier first.")
        exit(1)
    # 3. Model
    model = ClassifierGuidedGAN(latent_dim, dm.img_shape, clf, lr=lr)

    # 4. Train
    trainer = pl.Trainer(max_epochs=max_epochs)
    trainer.fit(model, dm)

    # 5. Save
    if save_path:
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")
    else:
        print("Model not saved. Provide a save path to save the model.")
    return model

def plot_mnist(model, save_path=None):
    latent_dim = model.latent_dim
    z = torch.randn(100, latent_dim, device=model.device)
    labels = torch.arange(10).repeat_interleave(10).to(model.device)
    imgs = model.generator(z, labels).cpu()
    grid = vutils.make_grid(imgs, nrow=10, normalize=True, value_range=(-1, 1))
    plt.figure(figsize=(10, 10))
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis('off')
    plt.title("Generated MNIST Samples")

    if save_path:
        save_path = os.path.join(save_path, "mnist_samples.png")
        vutils.save_image(grid, save_path)
        print(f"Generated images saved to {save_path}")
    else:
        plt.show()

def plot_cifar10(model, save_path=None):
    # Plot and save generated images for each class
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    num_classes = 10
    num_images_per_class = 10
    fig, axes = plt.subplots(10, num_images_per_class, figsize=(20, 20))  # Create a figure with subplots

    for class_idx in range(10):
        for i in range(num_images_per_class):
            # Generate an image
            with torch.no_grad():
                z = torch.randn(1, model.latent_dim, device=model.device)
                label_tensor = torch.tensor([class_idx], device=model.device)  # Use integer label for the class
                generated_img = model.generator(z, label_tensor).squeeze().cpu()

            # Denormalize and convert to NumPy array
            generated_img = generated_img.permute(1, 2, 0).numpy()
            generated_img = (generated_img * 0.5) + 0.5  # Denormalize
            generated_img = np.clip(generated_img, 0, 1)  # Clip to 0-1 range

            axes[class_idx, i].imshow(generated_img)
            axes[class_idx, i].axis("off")

    # Now add figure-level text, centered horizontally for each row
    for class_idx in range(num_classes):
        # Y position: depends on how the figure layout is spaced
        # We'll calculate a nice spot between the rows
        y = 1 - (class_idx) / num_classes
        fig.text(0.5, y, f'label: "{class_names[class_idx]}"', ha='center', va='center', fontsize=16, fontweight='bold')

    plt.tight_layout()

    plt.subplots_adjust(hspace=0.2)  # <<< More vertical space between rows!
    plt.suptitle("Generated CIFAR10 Samples", fontsize=20)

    if save_path:
        save_path = os.path.join(save_path, "cifar10_samples.png")
        plt.savefig(save_path)
        print(f"Generated images saved to {save_path}")
    else:
        plt.show()


def plot_samples(dataset_name, model, save_path=None):
    model.eval()

    with torch.no_grad():
        if dataset_name == "mnist":
            plot_mnist(model, save_path)
        elif dataset_name == "cifar10":
            plot_cifar10(model, save_path)
        else:
            print(f"Dataset {dataset_name} not supported for plotting samples.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['mnist', 'cifar10'], required=True)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--clf_path', type=str, default=None)
    parser.add_argument('--save_path', type=str, default=None)
    parser.add_argument('--plot_samples', action='store_true', help='Plot generated samples after training')
    args = parser.parse_args()
    # Train the CGAN
    model = train_cnn_cgan(args.dataset, args.batch_size, args.max_epochs, args.lr, args.clf_path, args.save_path)

    if args.plot_samples:
        # Plot samples
        plot_samples(args.dataset, model, args.save_path)

    if args.save_path:
        # Save the model
        torch.save(model.state_dict(), args.save_path)
        print(f"Model saved to {args.save_path}")
