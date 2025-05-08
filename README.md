# Classifier-Guided Conditional GAN (CGAN)

This project implements a Classifier-Guided Conditional Generative Adversarial Network (CGAN) capable of generating images for datasets like MNIST and CIFAR10. The CGAN uses a pre-trained classifier to guide the generator, improving the quality and coherence of the generated samples.

## Project Structure

```
.
├── classifier_guided_gan.py  # Defines the Generator, Discriminator, and CGAN (PyTorch Lightning module)
├── cnn_classifier.py         # Defines the CNN Classifier model (PyTorch Lightning module)
├── data_module.py            # Defines DataModules for classification and GAN training (MNIST, CIFAR10)
├── train_classifier.py       # Script to train the CNN classifier
├── train_cgan.py             # Script to train the Classifier-Guided CGAN
├── train.py                  # Main script to run the full pipeline (train classifier then CGAN)
├── requirements.txt          # Python dependencies
├── models/                   # Default directory to save trained model weights
├── plots/                    # Default directory to save training plots (loss, accuracy, confusion matrix)
├── samples/                  # Default directory to save generated image samples
├── data/                     # Default directory for downloading datasets
└── README.md                 # This file
```

## Components

-   **`cnn_classifier.py`**: Implements a Convolutional Neural Network (CNN) for image classification. This classifier is first trained on a target dataset (e.g., MNIST or CIFAR10).
-   **`classifier_guided_gan.py`**: Implements the core CGAN.
    -   `Generator`: A conditional generator network that takes a latent vector and a class label as input to produce an image.
    -   `Discriminator`: A conditional discriminator network that takes an image and a class label as input and tries to distinguish between real and fake images.
    -   `ClassifierGuidedGAN`: The main PyTorch Lightning module that orchestrates the training. It uses the pre-trained `CNNClassifier` to provide class guidance to the generator (via an additional loss term) and to infer labels for real images during discriminator training.
-   **`data_module.py`**: Contains two PyTorch Lightning `DataModule` classes:
    -   `ClassificationDataModule`: Handles data loading, transformations, and batching for training the `CNNClassifier`.
    -   `GANDataModule`: Handles data loading, transformations, and batching for training the `ClassifierGuidedGAN`.
-   **`train_classifier.py`**: A script to train the `CNNClassifier`. It can save the model weights and generate plots for training/validation loss, accuracy, and a confusion matrix.
-   **`train_cgan.py`**: A script to train the `ClassifierGuidedGAN`. It requires a path to a pre-trained classifier model and can save the trained GAN model and plot generated image samples.
-   **`train.py`**: The main training script that automates the process:
    1.  Trains the `CNNClassifier` using `train_classifier.py`.
    2.  Uses the trained classifier to train the `ClassifierGuidedGAN` using `train_cgan.py`.
    3.  Can save models and plots.

## Setup

1.  **Clone the repository (if applicable):**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-directory>
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    The `requirements.txt` file includes:
    ```
    matplotlib==3.10.0
    torch==2.6.0
    torchmetrics==1.7.0
    torchvision==0.21.0
    pytorch-lightning==2.5.1.post0
    numpy==2.0.2
    ```

## Running Instructions

The primary way to run the project is using `train.py`, which handles both classifier and CGAN training.

### Training the Full Pipeline (Classifier then CGAN)

Use the `train.py` script:

```bash
python train.py --dataset <dataset_name> [options]
```

**Arguments for `train.py`:**

-   `--dataset`: (Required) The dataset to use. Choices: `mnist`, `cifar10`.
-   `--batch_size`: Batch size for training (default: `64`).
-   `--max_epochs`: Number of epochs to train both the classifier and the GAN (default: `10`).
-   `--lr`: Learning rate for the optimizers (default: `2e-4` for GAN, `1e-3` for Classifier, `train.py` uses `2e-4` for classifier, you might want to adjust this or train separately for optimal classifier performance).
-   `--save_plots`: Action flag. If set, saves plots of training loss, accuracy, confusion matrix, and generated samples (default: `False`). Plots are saved to `plots/` and `samples/` directories respectively.
-   `--save_path`: Path to the directory to save the trained classifier and CGAN models (default: `models/`). Models will be saved as `<dataset_name>_classifier_params.pth` and `<dataset_name>_cgan_params.pth`.

**Example:**

```bash
python train.py --dataset mnist --max_epochs 20 --save_plots --save_path ./trained_models
```
This command will:
1. Train a CNN classifier on MNIST for 20 epochs.
2. Save the classifier model to `./trained_models/mnist_classifier_params.pth`.
3. Train a CGAN on MNIST for 20 epochs, using the just-trained classifier.
4. Save the CGAN model to `./trained_models/mnist_cgan_params.pth`.
5. Save training plots and generated samples to `plots/` and `samples/` respectively (these directories will be created if they don't exist).

### Training Only the Classifier

You can train the classifier separately using `train_classifier.py`:

```bash
python train_classifier.py --dataset <dataset_name> [options]
```

**Arguments for `train_classifier.py`:**

-   `--dataset`: (Required) Choices: `mnist`, `cifar10`.
-   `--save_path`: Path to save the trained classifier model (default: `models/<dataset_name>_classifier_params.pth`).
-   `--batch_size`: (default: `64`).
-   `--max_epochs`: (default: `10`).
-   `--lr`: Learning rate (default: `1e-3`).
-   `--plot_loss`: Action flag to plot training/validation loss.
-   `--plot_acc`: Action flag to plot training/validation accuracy.
-   `--confusion_matrix`: Action flag to plot the confusion matrix.
-   `--plot_path`: Directory to save plots if action flags are set (default: current directory, plots saved to `plots/` if `train.py` is used).

**Example:**

```bash
python train_classifier.py --dataset cifar10 --max_epochs 15 --lr 0.001 --save_path models/cifar_clf.pth --plot_loss --plot_acc --confusion_matrix --plot_path ./clf_reports
```

### Training Only the CGAN (with a pre-trained classifier)

You can train the CGAN separately using `train_cgan.py`, provided you have a trained classifier:

```bash
python train_cgan.py --dataset <dataset_name> --clf_path <path_to_classifier_model> [options]
```

**Arguments for `train_cgan.py`:**

-   `--dataset`: (Required) Choices: `mnist`, `cifar10`.
-   `--clf_path`: (Required) Path to the pre-trained classifier model (`.pth` file).
-   `--batch_size`: (default: `64`).
-   `--max_epochs`: (default: `10`).
-   `--lr`: Learning rate (default: `2e-4`).
-   `--save_path`: Path to save the trained CGAN model. If not provided, the model is not saved after training by this script directly (though `train.py` handles saving).
-   `--plot_samples`: Action flag to plot generated samples after training. Plots are saved to the `--save_path` directory if provided, otherwise shown.

**Example:**

```bash
python train_cgan.py --dataset cifar10 --clf_path models/cifar_clf.pth --max_epochs 50 --lr 0.0002 --save_path models/cifar_cgan.pth --plot_samples
```

## Notes

-   **Dataset Download**: The datasets (MNIST, CIFAR10) will be automatically downloaded to the `./data` directory if they are not found.
-   **Model Saving**:
    -   When using `train.py`, models are saved to the directory specified by `--save_path` (default: `models/`).
    -   `train_classifier.py` saves the model to `--save_path` (default: `models/<dataset_name>_classifier_params.pth`).
    -   `train_cgan.py` saves the model to `--save_path` only if specified.
-   **Classifier Performance**: The quality of the generated images from the CGAN is dependent on the performance of the pre-trained classifier. Ensure the classifier is well-trained. The `train.py` script uses the same learning rate and epochs for both classifier and GAN by default; you might achieve better classifier results by training it separately with fine-tuned hyperparameters (e.g., using `train_classifier.py` directly).
-   **Hardware**: Training GANs can be computationally intensive. Using a CUDA-enabled GPU or Apple Silicon (MPS) is highly recommended for faster training. The scripts should automatically detect and use available hardware.
-   **Plotting**: If `--save_plots` is used with `train.py`, or individual plot flags are used with `train_classifier.py` / `train_cgan.py`, ensure `matplotlib` is working correctly in your environment. Plots are saved to `plots/` and `samples/` (for generated images) or the directory specified by `--plot_path` / `--save_path`. 