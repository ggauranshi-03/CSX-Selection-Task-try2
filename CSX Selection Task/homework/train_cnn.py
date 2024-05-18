import torch
import torch.nn as nn
import torch.optim as optim
from models import CNNClassifier
from torch.utils.data import DataLoader
from torchvision import transforms
from utils import SuperTuxDataset, load_data

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define hyperparameters
learning_rate = 0.001
batch_size = 64
num_epochs = 50

# Data augmentation transforms
data_transforms = transforms.Compose(
    [
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
    ]
)

# Load training data
train_loader = load_data("data/train", transform=data_transforms, batch_size=batch_size)

# Initialize model
model = CNNClassifier().to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    # Print training statistics
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

# Save trained model
torch.save(model.state_dict(), "cnn_model.pth")
