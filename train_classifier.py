from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn as nn

from cnn_classifier import CNNClassifier

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load MNIST dataset
train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transforms.ToTensor())
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transforms.ToTensor())

train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)
test_loader = DataLoader(test_dataset, batch_size=64)

# Initialize the model, optimizer, and loss function
model = CNNClassifier().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

train_losses = []
val_losses = []

num_epochs = 10
for epoch in range(num_epochs):
    # Training loop
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0
    for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = output.max(1)
        train_total += target.size(0)
        train_correct += predicted.eq(target).sum().item()

    train_accuracy = train_correct / train_total
    avg_train_loss = train_loss / len(train_loader)

    # Validation loop
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss, acc = model.validation_step((data,target), 0)

            val_loss += loss.item()
            _, predicted = output.max(1)
            val_total += target.size(0)
            val_correct += predicted.eq(target).sum().item()

    val_accuracy = val_correct / val_total
    avg_val_loss = val_loss / len(val_loader)

    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)

    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

# Test loop
model.eval()
test_loss = 0
test_correct = 0
test_total = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss, acc = model.test_step((data, target), 0)

        test_loss += loss.item()
        _, predicted = output.max(1)
        test_total += target.size(0)
        test_correct += predicted.eq(target).sum().item()

test_accuracy = test_correct / test_total
avg_test_loss = test_loss / len(test_loader)
print(f"Test Loss: {avg_test_loss:.4f}, Test Acc: {test_accuracy:.4f}")

# Plot train and validation loss
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig('plots/cnn_losses.png')


torch.save(model.state_dict(), 'models/cnn_model_params.pth')
# prompt: Take random 10 samples from the test dataset for each label (0 to 9), and plot the samples for each label in a small row, with the model prediction for each sample label

# load the model parameters from the saved file 'models/cnn_model_params.pth'
# model = CNNClassifier().to(device)
# model.load_state_dict(torch.load('models/cnn_model_params.pth'))

# Get random samples for each label
num_samples_per_label = 10
label_samples = {i: [] for i in range(10)}

for data, target in test_loader:
  for i in range(len(target)):
    label = target[i].item()
    if len(label_samples[label]) < num_samples_per_label:
      label_samples[label].append((data[i], target[i]))


# Plot samples for each label in the same plot, row by row

fig, axs = plt.subplots(10, num_samples_per_label, figsize=(16, 16))

for label in range(10):
    for i in range(num_samples_per_label):
        image, target = label_samples[label][i]
        image = image.squeeze().cpu().numpy()
        axs[label, i].imshow(image, cmap="gray")
        axs[label, i].axis("off")

        model.eval()
        with torch.no_grad():
            image = image.reshape(1, 1, 28, 28)
            image = torch.from_numpy(image).to(device)
            output = model(image)
            predicted_label = torch.argmax(output, dim=1).item()
            axs[label, i].set_title(f"Pred: {predicted_label}")

    plt.savefig('plots/mnist_samples.png')
  # fig.suptitle(f"Label: {label}")
  # for i in range(num_samples_per_label):
  #     image, target = label_samples[label][i]
  #     image = image.squeeze().cpu().numpy()
  #     axs[i].imshow(image, cmap="gray")
  #     axs[i].axis("off")
  #
  #     model.eval()
  #     with torch.no_grad():
  #         image = image.reshape(1, 1, 28, 28)
  #         image = torch.from_numpy(image).to(device)
  #         output = model(image)
  #         predicted_label = torch.argmax(output, dim=1).item()
  #         axs[i].set_title(f"Pred: {predicted_label}")
  #
  # plt.show()


# prompt: Save the model's trained parameters

