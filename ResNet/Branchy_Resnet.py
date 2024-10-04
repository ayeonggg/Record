import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils import data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
import numpy as np 

# define pytorch device - useful for device-agnostic execution
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# define model parameters
NUM_EPOCHS = 1000
BATCH_SIZE = 128
IMAGE_DIM = 32  # pixels
NUM_CLASSES = 10  # 10 classes for CIFAR-10 dataset
DEVICE_IDS = [0]  # GPUs to use
OUTPUT_DIR = 'resnet_model'
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, 'models')  # model checkpoints

# make checkpoint path directory
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=NUM_CLASSES):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

def ResNet18(num_classes=NUM_CLASSES):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)

if __name__ == '__main__':
    # create model
    resnet = ResNet18(num_classes=NUM_CLASSES).to(device)
    # train on multiple GPUs
    resnet = torch.nn.parallel.DataParallel(resnet, device_ids=DEVICE_IDS)
    print(resnet)
    print('ResNet created')

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(IMAGE_DIM, scale=(0.9, 1.0), ratio=(0.9, 1.1)),
        transforms.CenterCrop(IMAGE_DIM),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
    
    traindata = datasets.CIFAR10(root='./CIFAR', train=True, transform=transform, download=True)
    
    testtransform = transforms.Compose([
        transforms.CenterCrop(IMAGE_DIM),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
    
    testdata = datasets.CIFAR10(root='./CIFAR', train=False, transform=testtransform, download=True)
    print('Dataset created')
    
    dataloader = data.DataLoader(
        traindata,
        shuffle=True,
        pin_memory=True,
        num_workers=8,
        drop_last=True,
        batch_size=BATCH_SIZE)
    print('Dataloader created')

    optimizer = optim.Adam(params=resnet.parameters(), lr=0.0001)
    print('Optimizer created')

    # multiply LR by 1 / 10 after every 30 epochs
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    print('LR Scheduler created')

    # start training!!
    print('Starting training...')
    resnet.train()
    total_steps = 1
    end = False
    for epoch in range(NUM_EPOCHS):
        for imgs, classes in dataloader:
            imgs, classes = imgs.to(device), classes.to(device)
            optimizer.zero_grad()
            # calculate the loss
            output = resnet(imgs)
            loss = F.cross_entropy(output, classes)

            # update the parameters
            loss.backward()
            optimizer.step()

            # log the information and add to tensorboard
            if total_steps % 50 == 0:
                _, preds = torch.max(output, 1)
                accuracy = torch.sum(preds == classes)

                print('Epoch: {} \tStep: {} \tLoss: {:.4f} \tAcc: {}'
                      .format(epoch + 1, total_steps, loss.item(), accuracy.item()))

            if total_steps % 300 == 0:
                # Validation
                valdataloader = data.DataLoader(
                    testdata,
                    shuffle=True,
                    pin_memory=True,
                    num_workers=8,
                    drop_last=True,
                    batch_size=128)
                
                correct_count = 0
                total_count = 0
                resnet.eval()
                for images, labels in valdataloader:
                    images, labels = images.to(device), labels.to(device)
                    with torch.no_grad():  # no gradient descent!
                        logps = resnet(images)

                    ps = torch.exp(logps)
                    prob = ps.cpu().numpy()
                    preds = np.argmax(prob, axis=1)
                    correct_count += np.sum(preds == labels.cpu().numpy())
                    total_count += labels.size(0)

                print("Number Of Images Tested =", total_count)
                print("\nModel Accuracy =", (correct_count / total_count))
                
                if correct_count / total_count > 0.95:
                    end = True
                
                resnet.train()
            
            if end:
                break

            total_steps += 1
        
        if end:
            break
        
        lr_scheduler.step()

        if epoch % 100 == 9:
            torch.save(resnet.state_dict(), os.path.join(CHECKPOINT_DIR, f'epoch_{epoch+1}_model.pt'))
