# Script that:
# 1. Trains a classifier on MNIST or CIFAR10
# 2. uses the classifier model train a CGAN on the same dataset

import argparse
import os

from train_cgan import train_cnn_cgan, plot_samples
from train_classifier import train_cnn_classifier, plot_losses, plot_accuracy, plot_confusion_matrix

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a CGAN with a pre-trained classifier.")
    parser.add_argument('--dataset', type=str, choices=['mnist', 'cifar10'], required=True, help='Dataset to use')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--max_epochs', type=int, default=10, help='Number of epochs to train the GAN')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate for the optimizer')
    parser.add_argument('--save_plots', action='store_true', default=False, help='Save plots of training loss and accuracy')
    parser.add_argument('--save_path', type=str, default="models", help='Path to dir to save the classifier + cgan models')

    args = parser.parse_args()
    os.makedirs(args.save_path, exist_ok=True)

    classifier_save_path = os.path.join(args.save_path, f"{args.dataset}_classifier_params.pth")

    if args.save_plots:
        os.makedirs("plots", exist_ok=True)
        os.makedirs("samples", exist_ok=True)


    # Train Classifier
    classifier, train_loss, train_acc, val_loss, val_acc, dm = train_cnn_classifier(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        lr=args.lr,
        save_path=classifier_save_path
    )

    print(f"Classifier trained on {args.dataset} dataset.")
    print(f"Train Loss: {train_loss[-1]}, Train Accuracy: {train_acc[-1]}")
    print(f"Validation Loss: {val_loss[-1]}, Validation Accuracy: {val_acc[-1]}")
    print(f"Classifier model weights saved to {classifier_save_path}")

    if args.save_plots:
        plots_dir = "plots"
        # Save plots of training loss and accuracy
        plot_losses(args.dataset, train_loss, val_loss, save_path=plots_dir)
        plot_accuracy(args.dataset, train_acc, val_acc, save_path=plots_dir)
        plot_confusion_matrix(args.dataset, classifier, dm, save_path=plots_dir)


    # Train CGAN
    cgan_save_path = os.path.join(args.save_path, f"{args.dataset}_cgan_params.pth")
    train_cnn_cgan(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        lr=args.lr,
        clf_path=classifier_save_path,
        save_path=cgan_save_path
    )
    print(f"CGAN trained on {args.dataset} dataset.")
    print(f"CGAN model weights saved to {cgan_save_path}")

    if args.save_plots:
        samples_dir = "samples"
        # Save plots of generated samples
        plot_samples(args.dataset, cgan_save_path, save_path=samples_dir)
        print(f"Generated samples saved to {samples_dir}")
