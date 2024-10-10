import torch
import torch.nn as nn
import thriftpy2
import numpy as np
from thriftpy2.rpc import make_server

# NetExit3Part2 모델 클래스 정의
class NetExit3Part2L(nn.Module):
    def __init__(self, num_classes=10):
        super(NetExit3Part2L, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, padding=3)  # 커널 크기 7x7
        self.relu = nn.ReLU()
        self.norm = nn.LocalResponseNorm(3)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.norm(x)
        x = self.pool(x)
        return x

# GPU에서 학습된 모델 로드
model_path = "/home/srlab/.vscode-server/ayeong/resnet_data_out/checkpoints/epoch_1000_model.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NetExit3Part2L().to(device)
model.load_state_dict(torch.load(model_path, map_location=device), strict=False)

# 추론 함수
def run_inference(data):
    data = torch.tensor(data).to(device)
    model.eval()
    with torch.no_grad():
        output = model(data)
    return output.cpu().numpy()

# 정확도 계산 함수
def calculate_accuracy(predictions, labels):
    predicted_labels = np.argmax(predictions, axis=1)
    correct = (predicted_labels == labels).sum()
    accuracy = correct / len(labels)
    return accuracy

# Partition 관련 Thrift 파일 로드
partition_thrift = thriftpy2.load("/home/srlab/.vscode-server/ayeong/partition.thrift", module_name="partition_thrift")

# Thrift 서버 핸들러 클래스 정의
class PartitionHandler:
    def partition(self, inputs, labels):
        print(f"Received input: {inputs}")
        predictions = run_inference(inputs)
        
        # 예시로 predictions가 2차원 배열일 때, 레이블도 1차원 배열이어야 함
        if len(predictions) == len(labels):
            accuracy = calculate_accuracy(predictions, labels)
            print(f"Accuracy: {accuracy * 100:.2f}%")  # 정확도 출력
        else:
            print("Error: Number of predictions and labels do not match.")
        
        return predictions.tolist()
# Thrift 서버 실행
server = make_server(partition_thrift.Partition, PartitionHandler(), '0.0.0.0', 8080)
print("서버가 실행 중입니다...")
server.serve()
