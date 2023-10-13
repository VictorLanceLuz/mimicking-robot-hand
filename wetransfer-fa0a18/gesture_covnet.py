import torch
from torch import nn
from torch import optim
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F 
from torchvision import datasets, transforms, models
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
startTime = datetime.now()

# Parameters
TRAINED_PATH = 'trained.pth'    # Path to place trained model
DATA_DIR = 'hands'
VALIDATION_SIZE = 0.2   # Use 20% of the data as validation, rest for training
IMAGE_SIZE = 100        # Transform all images to (IMAGE_SIZE, IMAGE_SIZE)
BATCH = 64              # Number examples per iteration
EPOCHS = 5             # Number of times it goes through training dataset
LEARNING_RATE = 0.002

steps = 0
running_loss = 0
print_per = 10

# Use GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pretrained model
model = models.resnet50(pretrained=True)

# Functions
# Sets up and loads the data into the training and test datasets
def data_loader(directory, valid):
    train_transforms = transforms.Compose([transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
                                            transforms.ToTensor(),
                                            ])
    test_transforms = transforms.Compose([transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
                                            transforms.ToTensor(),
                                            ])
    train_data = datasets.ImageFolder(directory, transform=train_transforms)
    test_data = datasets.ImageFolder(directory, transform=test_transforms)

    training_len = len(train_data)
    indices = list(range(training_len))
    split = int(np.floor(valid*training_len)) # Split the dataset
    np.random.shuffle(indices)                   # Randomize to even distribution
    train_index, test_index = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_index)
    test_sampler = SubsetRandomSampler(test_index)
    train_loader = torch.utils.data.DataLoader(train_data, sampler=train_sampler, batch_size=BATCH)
    test_loader = torch.utils.data.DataLoader(test_data, sampler=test_sampler, batch_size=BATCH)

    return train_loader, test_loader

#Load the training and test loaders
train_loader, test_loader = data_loader(DATA_DIR, VALIDATION_SIZE)
print(train_loader.dataset.classes)     # Display which classes were identified

# Trains the model
def train(num_epochs, step, run_loss, print_each):
    train_losses, test_losses = [], []
    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            step += 1
            inputs, labels = inputs.to(device), labels.to(device)   # Puts into GPU
            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = loss_function(logps, labels)
            loss.backward()
            optimizer.step()
            run_loss += loss.item()

            if step % print_each == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()

                # Validation
                with torch.no_grad():
                    for inputs, labels in test_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = loss_function(logps, labels)
                        valid_loss += batch_loss.item()

                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                
                train_losses.append(run_loss/len(train_loader))
                test_losses.append(valid_loss/len(test_loader))
                print(f"Epoch {epoch+1}/{num_epochs}.. "
                  f"Train loss: {run_loss/print_each:.3f}.. "
                  f"Test loss: {valid_loss/len(test_loader):.3f}.. "
                  f"Test accuracy: {accuracy/len(test_loader):.3f}")

            run_loss = 0
            model.train()
    torch.save(model, TRAINED_PATH)

    return train_losses, test_losses

# Freese pretrained layers to avoid back propagation
for param in model.parameters():
    param.requires_grad = False

# Redefine final fully connected layer
model.fc = nn.Sequential(nn.Linear(2048, 512),
                                nn.ReLU(),
                                nn.Dropout(0.2),
                                nn.Linear(512,10),
                                nn.LogSoftmax(dim=1))

loss_function = nn.NLLLoss()    # defining loss function
optimizer = optim.Adam(model.fc.parameters(), lr = LEARNING_RATE) # Using Adam optimizer
model.to(device)

train_loss, test_loss = train(EPOCHS, steps, running_loss, print_per) # Training the model
print(datetime.now()-startTime)

# Plot the training and validation loss
plt.plot(train_loss, label='Training Loss')
plt.plot(test_loss, label='Validation Loss')
plt.legend(frameon=False)
plt.show()
