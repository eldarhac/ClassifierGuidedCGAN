import os.path

from pytorch_lightning import Trainer
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import confusion_matrix
import numpy as np
import argparse

from cnn_classifier import CNNClassifier, HistoryCallback
from data_module import ClassificationDataModule

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


def train_cnn_classifier(dataset_name, batch_size=64, max_epochs=10, lr=1e-3, save_path=None):
    # Prepare Data
    dm = ClassificationDataModule(dataset_name=dataset_name, batch_size=batch_size)
    dm.setup()

    # Train
    classifier = CNNClassifier(in_channels=dm.in_channels, num_classes=10, lr=lr)
    history_cb = HistoryCallback()
    mnist_trainer = Trainer(max_epochs=max_epochs, num_sanity_val_steps=0, callbacks=[history_cb], devices=1)
    mnist_trainer.fit(classifier, dm)

    train_loss = classifier.train_losses
    train_acc = classifier.train_accs

    mnist_trainer.validate(classifier, dm)

    val_loss = classifier.val_losses[1:]
    val_acc = classifier.val_accs[1:]

    if save_path:
        torch.save(classifier.state_dict(), save_path)
        print(f"Model saved to {save_path}")

    return classifier, train_loss, train_acc, val_loss, val_acc, dm


def plot_losses(dataset_name, train_loss, val_loss, save_path=None):
    epochs = [i for i in range(1, len(train_loss) + 1)]
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")
    plt.title(f"{dataset_name.upper()} Classifier Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    if save_path:
        save_path_loss = os.path.join(save_path, f"{dataset_name}_classifier_loss_plot.png")
        plt.savefig(save_path_loss)
        print(f"Loss plot saved to {save_path}")
    else:
        plt.show()


def plot_accuracy(dataset_name, train_acc, val_acc, save_path=None):
    epochs = [i for i in range(1, len(train_acc) + 1)]
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_acc, label="Train Accuracy")
    plt.plot(epochs, val_acc, label="Validation Accuracy")
    plt.title(f"{dataset_name.upper()} Classifier Training & Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    if save_path:
        save_path_acc = os.path.join(save_path, f"{dataset_name}_classifier_accuracy_plot.png")
        plt.savefig(save_path_acc)
        print(f"Accuracy plot saved to {save_path}")
    else:
        plt.show()


def plot_confusion_matrix(dataset_name, classifier, dm, save_path=None):
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in dm.val_dataloader():
            x = x.to(classifier.device)
            logits = classifier(x)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())

    cm = confusion_matrix(all_labels, all_preds)

    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=np.arange(10),
           yticklabels=np.arange(10),
           title=f'{dataset_name.upper()} Classifier Confusion Matrix',
           ylabel='True Label',
           xlabel='Predicted Label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    if save_path:
        save_path_cm = os.path.join(save_path, f"{dataset_name}_confusion_matrix.png")
        plt.savefig(save_path_cm, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")

    else:
        fig.tight_layout()
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['mnist', 'cifar10'], required=True)
    parser.add_argument('--save_path', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--plot_loss', action='store_true', default=False)
    parser.add_argument('--plot_acc', action='store_true', default=False)
    parser.add_argument('--confusion_matrix', action='store_true', default=False)
    parser.add_argument('--plot_path', type=str, default=None)
    args = parser.parse_args()

    save_path = args.save_path
    if save_path is None:
        save_path = os.path.join("models", f"{args.dataset}_classifier_params.pth")

    classifier, train_loss, train_acc, val_loss, val_acc, dm = train_cnn_classifier(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        lr=args.lr,
        save_path=save_path
    )
    print(f"Classifier trained on {args.dataset} dataset.")
    print(f"Train Loss: {train_loss[-1]}, Train Accuracy: {train_acc[-1]}")
    print(f"Validation Loss: {val_loss[-1]}, Validation Accuracy: {val_acc[-1]}")
    print(f"Model weights saved to {args.save_path}")
    if args.plot_loss:
        plot_losses(args.dataset, train_loss, val_loss, save_path=args.plot_path)
    if args.plot_acc:
        plot_accuracy(args.dataset, train_acc, val_acc, save_path=args.plot_path)
    if args.confusion_matrix:
        plot_confusion_matrix(args.dataset, classifier, dm, save_path=args.plot_path)