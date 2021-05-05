import torchvision
import torch
import torchvision.transforms as transforms
import os
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim
import scipy

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

train_dataset_path=r'D:\Users\emill\testTrainWeebs\newtrain\newimg'
test_dataset_path=r"D:\Users\emill\testTrainWeebs\test"
#custom_dataset_path=r'C:\Users\emill\PycharmProjects\canscanner/custom'

train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                      transforms.RandomRotation(20),
                                      transforms.ToTensor()])

test_transforms = transforms.Compose([transforms.ToTensor()])


train_dataset=torchvision.datasets.ImageFolder(root=train_dataset_path,transform=train_transforms)
test_dataset=torchvision.datasets.ImageFolder(root=test_dataset_path,transform=test_transforms)
#custom_dataset=torchvision.datasets.ImageFolder(root=custom_dataset_path,transform=test_transforms)


def show_transformed_images(dataset):
    loader = torch.utils.data.DataLoader(dataset, batch_size=6, shuffle=True)
    batch = next(iter(loader))
    images, labels = batch

    grid = torchvision.utils.make_grid(images, nrow=3)
    plt.figure(figsize=(11, 11))
    plt.imshow(np.transpose(grid, (1, 2, 0)))
    print(labels)


train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
#custom_loader = torch.utils.data.DataLoader(custom_dataset, batch_size=1, shuffle=False)

learning_rate=0.0001
num_epochs = 3


model = torchvision.models.googlenet(pretrained=True)
num_ftrs = model.fc.in_features

model.fc=nn.Linear(num_ftrs,2)
model.to(device)
#model = HotDogClassifier()
criterion=nn.CrossEntropyLoss()
#criterion=nn.BCELoss()
optimizer = optim.Adam(model.parameters(),lr=learning_rate)

def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)




    model.train()
    return float(num_correct)/float(num_samples)*100

for epoch in range(num_epochs):
    losses = []

    for batch_idx, (data, targets) in enumerate(train_loader):
        # Get data to cuda if possible
        data=data.to(device=device)
        targets=targets.to(device=device)
        # forward
        scores = model(data)
        loss = criterion(scores, targets)

        losses.append(loss.item())

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()
    val_acc=check_accuracy(test_loader, model)

    print(f"Cost at epoch {epoch} is {sum(losses)/len(losses)}, Validation acc:{val_acc}")

# Check accuracy on training to see how good our model is





def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(
            f"Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}"
        )

    model.train()

print("Checking accuracy on Training Set")
check_accuracy(train_loader, model)

print("Checking accuracy on Test Set")
check_accuracy(test_loader, model)




torch.save(model, "weebmodel3.pt")