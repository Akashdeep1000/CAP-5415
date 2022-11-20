import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import CNN_model_v2
from torch import optim
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

# check if CUDA is available
device = ("cuda" if torch.cuda.is_available() else "cpu")
print(device)




num_workers = 0
batch_size = 20
# validation set out of training set
valid_size = 0.2
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

#training and test datasets
train_data = datasets.CIFAR10('data', train=True,
                              download=True, transform=transform)
test_data = datasets.CIFAR10('data', train=False,
                             download=True, transform=transform)

# obtain training indices that will be used for validation
num_train = len(train_data)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(valid_size * num_train))
train_idx, valid_idx = indices[split:], indices[:split]

# define samplers for obtaining training and validation batches
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

# prepare data loaders (combine dataset and sampler)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
    sampler=train_sampler, num_workers=num_workers)
valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, 
    sampler=valid_sampler, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, 
    num_workers=num_workers)

# specify the image classes
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']


model = CNN_model_v2.CNN_v2()
if torch.cuda.is_available():
    model.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=.01)
num_correct = 0
num_samples = 0
num_correct_train =0

n_epochs = 30

#List to store loss an accuracy for training and testing to visualize
train_losslist = []
validation_losslist =[]
val_accuracy_list =[]
train_accuracy_list =[]
valid_loss_min = np.Inf # track change in validation loss

for epoch in range(1, n_epochs+1):

    train_loss = 0.0
    valid_loss = 0.0
    

#training Start
    for data, target in train_loader:

        
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        output = model(data)

        loss = criterion(output, target)

        loss.backward()

        optimizer.step()
        _,predictions = output.max(1)
        num_correct_train = (predictions==target).sum()
        running_acc = float(num_correct_train)/float(data.shape[0])

        train_loss += loss.item()*data.size(0)
        
    
    # validate the model 

    model.eval()
    for data, target in valid_loader:
        data, target = data.to(device), target.to(device)
        
        output = model(data)
        _,predictions = output.max(1)

        loss = criterion(output, target)
        valid_loss += loss.item()*data.size(0)
        
        num_correct += (predictions==target).sum()
        num_samples += predictions.size(0)
        acc = float(num_correct)/float(num_samples)

    train_loss = train_loss/len(train_loader.dataset)
    valid_loss = valid_loss/len(valid_loader.dataset)
    train_losslist.append(train_loss)
    validation_losslist.append(valid_loss)
    val_accuracy_list.append(acc)
    train_accuracy_list.append(running_acc)

        
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch, train_loss, valid_loss))
    
    # save model if validation loss has decreased
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
        valid_loss_min,
        valid_loss))
        torch.save(model.state_dict(), 'cifar_v2.pt')
        valid_loss_min = valid_loss

print(f"Test accuracy after epoch {epoch}: {100*acc}")
plt.plot(range(n_epochs), train_losslist)
plt.xlabel("Epoch")
plt.ylabel("Training_Loss")
plt.title("Performance of Model 1")
plt.show()

plt.plot(range(n_epochs), validation_losslist)
plt.xlabel("Epoch")
plt.ylabel("Validation_Loss")
plt.title("Performance of Model 1")
plt.show()

plt.plot(range(n_epochs), val_accuracy_list)
plt.xlabel("Epoch")
plt.ylabel("Validation_Accuracy")
plt.title("Performance of Model 1")
plt.show()

plt.plot(range(n_epochs), train_accuracy_list)
plt.xlabel("Epoch")
plt.ylabel("Training_Accuracy")
plt.title("Performance of Model 1")
plt.show()