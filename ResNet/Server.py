from flask import Flask, request, jsonify
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import numpy as np

app = Flask(__name__)

NUM_CLASSES = 10  # CIFAR-10 데이터셋의 클래스 수
model_path = '/home/ayeong/resnet_model/models/epoch_910_model.pt'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        layers = [block(self.in_channels, out_channels, stride, downsample)]
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

def remove_module_prefix(state_dict):
    return {key.replace("module.", ""): value for key, value in state_dict.items()}

model = ResNet18(num_classes=NUM_CLASSES)
state_dict = torch.load(model_path, map_location=device)
state_dict = remove_module_prefix(state_dict)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    img = Image.open(file.stream).convert('RGB')
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img)
        probabilities = torch.softmax(output, dim=1).cpu().numpy()

    return jsonify({'prediction': int(np.argmax(probabilities)), 'probabilities': probabilities.tolist()})

@app.route('/accuracy', methods=['POST'])
def calculate_accuracy():
    if 'images' not in request.files or 'labels' not in request.form:
        return jsonify({'error': 'No data provided'}), 400

    images = request.files.getlist('images')
    labels = request.form.getlist('labels')
    correct_count = 0
    total_count = 0

    for img_file, label in zip(images, labels):
        img = Image.open(img_file.stream).convert('RGB')
        img = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img)
            probabilities = torch.softmax(output, dim=1).cpu().numpy()
            predicted_label = int(np.argmax(probabilities))

        if predicted_label == int(label):
            correct_count += 1
        total_count += 1

    accuracy = correct_count / total_count if total_count > 0 else 0
    return jsonify({'accuracy': accuracy})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
