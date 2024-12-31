import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import os
from torch.utils.data import DataLoader

# 사전 학습된 ResNet 모델 불러오기
resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# GPU가 있으면 GPU로 설정하고, 그렇지 않으면 CPU로 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resnet.to(device)

# 모델을 평가 모드로 전환
resnet.eval()

# 필요한 변수들 초기화
correct_count = 0
total_count = 0
epoch = 0
end = False

# 디렉토리 경로 설정
CHECKPOINT_DIR = '/home/srlab/.vscode-server/ayeong/resnet_data_out/checkpoints'
PARAM_PATH = '/home/srlab/.vscode-server/ayeong/resnet_data_out/models/'

# 파트 이름 정의
netParts = ['NetExit1Part1', 'NetExit1Part2', 'NetExit2Part1', 'NetExit2Part2', 'NetExit3Part1', 'NetExit3Part2', 'NetExit4Part1', 'NetExit4Part2']

# 디렉토리 생성 (없을 경우)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(PARAM_PATH, exist_ok=True)

# 데이터셋 로딩 및 변환 설정
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet 입력 크기
    transforms.ToTensor(),
])
train_dataset = datasets.CIFAR10(root='/home/srlab/DNN-Partition-demo/CIFAR/cifar-10-batches-py', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 학습 루프
max_epochs = 500
while epoch < max_epochs:
    resnet.train()  # 모델을 학습 모드로 전환
    correct_count = 0
    total_count = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # 모델의 순전파
        outputs = resnet(inputs)
        _, predicted = torch.max(outputs, 1)

        total_count += labels.size(0)
        correct_count += (predicted == labels).sum().item()

    # 정확도 계산
    if total_count > 0:
        accuracy = correct_count / total_count
        print(f"\nEpoch [{epoch + 1}/{max_epochs}] - Model Accuracy = {accuracy:.4f}")

        # 모델 저장 경로 설정 및 저장 (정확도와 관계없이 매 에포크마다 저장)
        for i, netPart in enumerate(netParts):
            save_path = os.path.join(PARAM_PATH, f'{netPart}_epoch_{epoch + 1}.pth')
            torch.save(resnet.state_dict(), save_path)
            print(f'{netPart} 모델 저장됨: {save_path}')

    # 100 에포크마다 체크포인트 저장
    if (epoch + 1) % 100 == 0:
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f'epoch_{epoch + 1}_model.pt')
        torch.save(resnet.state_dict(), checkpoint_path)
        print(f'Checkpoint saved at epoch {epoch + 1}')

    epoch += 1  # 에포크 증가

print("학습 완료.")
