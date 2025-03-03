import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import tqdm 

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
    ])  

# Load the STL-10 dataset
train_dataset = datasets.STL10(root='data', split='train',download=True, transform=transform)
test_dataset = datasets.STL10(root='data', split='test',download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64,shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64,shuffle=False)

# Load pre-trained AlexNet model
alexnet = models.alexnet(pretrained=True)
alexnet.classifier[6] = nn.Linear(alexnet.classifier[6].in_features, 10)

# Move the model to the GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else
"cpu")
alexnet.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(alexnet.parameters(), lr=0.001)

def train(model, device, train_loader, criterion, optimizer,epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx *len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.6f}')

def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            
    test_loss /= len(test_loader.dataset)
    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy:{correct}/{len(test_loader.dataset)} ({100. * correct /len(test_loader.dataset):.2f}%)')

# Training and testing the model# Training and testing the model
num_epochs = 45
for epoch in range(1, num_epochs + 1):
    train(alexnet, device, train_loader, criterion, optimizer,epoch)
    test(alexnet, device, test_loader, criterion)
