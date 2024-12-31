import torch.nn as nn
import torchvision.models as models

NUM_CLASSES = 10  # CIFAR-10의 경우 10개의 클래스

####################
# Exit 1 Part 1
####################
class ResNetExit1Part1L(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        resnet = models.resnet18(pretrained=False)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1  # 첫 번째 Residual Block
        self.mainConvLayers = [self.conv1, self.layer1]

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        return x

####################
# Exit 1 Part 1 Right
####################
class ResNetExit1Part1R(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        resnet = models.resnet18(pretrained=False)
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.mean([2, 3])  # Adaptive average pooling
        x = self.fc(x)
        return x

####################
# Exit 2 Part 1
####################
class ResNetExit2Part1L(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        resnet = models.resnet18(pretrained=False)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2  # 첫 번째 두 개의 Residual Block까지
        self.mainConvLayers = [self.layer1, self.layer2]

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

####################
# Exit 2 Part 1 Right
####################
class ResNetExit2Part1R(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        resnet = models.resnet18(pretrained=False)
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.mean([2, 3])  # Adaptive average pooling
        x = self.fc(x)
        return x

####################
# Exit 3 Part 1
####################
class ResNetExit3Part1L(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        resnet = models.resnet18(pretrained=False)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3  # 세 번째 Residual Block까지
        self.mainConvLayers = [self.layer1, self.layer2, self.layer3]

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

####################
# Exit 3 Part 1 Right
####################
class ResNetExit3Part1R(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        resnet = models.resnet18(pretrained=False)
        self.layer4 = resnet.layer4
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.layer4(x)
        x = x.mean([2, 3])  # Adaptive average pooling
        x = self.fc(x)
        return x

####################
# Exit 4 Part 1
####################
class ResNetExit4Part1L(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        resnet = models.resnet18(pretrained=False)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4  # 전체 레이어까지
        self.mainConvLayers = [self.layer1, self.layer2, self.layer3, self.layer4]

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

####################
# Exit 4 Part 1 Right
####################
class ResNetExit4Part1R(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        resnet = models.resnet18(pretrained=False)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = x.mean([2, 3])  # Adaptive average pooling
        x = self.fc(x)
        return x

####################
# Model Pair
####################
NetExit1Part1 = [ResNetExit1Part1L, ResNetExit1Part1R]
NetExit2Part1 = [ResNetExit2Part1L, ResNetExit2Part1R]
NetExit3Part1 = [ResNetExit3Part1L, ResNetExit3Part1R]
NetExit4Part1 = [ResNetExit4Part1L, ResNetExit4Part1R]
