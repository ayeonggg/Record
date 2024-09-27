import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import os
from torch.utils.data import DataLoader

# 사전 학습된 AlexNet 모델 불러오기
alexnet = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
alexnet.load_state_dict(torch.load('/home/user/DNN-Partition-demo/alexnet_data_out/models/NetExit3Part2_L.pth', weights_only=True))
# 모델을 CPU로 설정
device = torch.device('cpu')
alexnet.to(device)

# 모델을 평가 모드로 전환
alexnet.eval()

# 필요한 변수들 초기화
correct_count = 0
total_count = 0
epoch = 0
end = False

# 디렉토리 경로 설정
CHECKPOINT_DIR = '/home/user/DNN-Partition-demo/alexnet_data_out/checkpoints'
PARAM_PATH = '/home/user/DNN-Partition-demo/alexnet_data_out/models/'
netPair = 'NetExit3Part2'

# 디렉토리 생성 (없을 경우)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(PARAM_PATH, exist_ok=True)

# 데이터셋 로딩 및 변환 설정
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # AlexNet 입력 크기
    transforms.ToTensor(),
])
train_dataset = datasets.CIFAR10(root='/home/user/DNN-Partition-demo/CIFAR/cifar-10-batches-py', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 학습 루프 (최대 1000 에포크)
max_epochs = 1000
while epoch < max_epochs:
    alexnet.train()  # 모델을 학습 모드로 전환
    correct_count = 0
    total_count = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # 모델의 순전파
        outputs = alexnet(inputs)
        _, predicted = torch.max(outputs, 1)

        total_count += labels.size(0)
        correct_count += (predicted == labels).sum().item()

    # 정확도 계산
    if total_count > 0:
        accuracy = correct_count / total_count
        print(f"\nEpoch [{epoch + 1}/{max_epochs}] - Model Accuracy = {accuracy:.4f}")

        # 모델 저장 경로 설정 및 저장
        save_path = os.path.join(PARAM_PATH, f'model_epoch_{epoch + 1}.pth')
        torch.save(alexnet.state_dict(), save_path)
        print(f'Model saved at {save_path}')

        # 정확도가 95%를 넘으면 학습 중단
        if accuracy > 0.95:
            end = True

    # 100 에포크마다 체크포인트 저장
    if (epoch + 1) % 100 == 0:
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f'epoch_{epoch + 1}_model.pt')
        torch.save(alexnet.state_dict(), checkpoint_path)
        torch.save(alexnet.state_dict(), os.path.join(PARAM_PATH, f'{netPair}_L.pth'))
        print(f'Checkpoint saved at epoch {epoch + 1}')

    epoch += 1  # 에포크 증가

print("학습 완료.")

