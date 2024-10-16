import torch
import torch.nn as nn
import thriftpy2
import numpy as np
from thriftpy2.rpc import make_server

# NetExit3Part2 모델 클래스 정의
class NetExit3Part2L(nn.Module):
    def __init__(self, num_classes=10):
        super(NetExit3Part2L, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, padding=3)
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
model_path = "/home/srlab/.vscode-server/ayeong/resnet_data_out/models/NetExit3Part2_L.pth"
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
    # 예측된 레이블 계산
    predicted_labels = np.argmax(predictions, axis=1)
    
    # 레이블이 1차원 배열인지 확인
    labels = np.array(labels)
    if len(labels.shape) > 1:
        labels = np.argmax(labels, axis=1)
    
    # 예측과 레이블의 길이가 동일한지 확인
    if len(predicted_labels) != len(labels):
        print("Error: Predictions and labels lengths do not match.")
        return 0.0
    
    # 정확도 계산
    correct = (predicted_labels == labels).sum()
    accuracy = correct / len(labels)
    return accuracy
partition_thrift = thriftpy2.load("/home/srlab/.vscode-server/ayeong/partition.thrift", module_name="partition_thrift")

# Thrift 서버 핸들러 클래스 정의
class PartitionHandler:
    def partition(self, inputs, labels):
        print(f"Received input: {inputs}")
        predictions = run_inference(inputs)
        
        # 정확도 계산
        accuracy = calculate_accuracy(predictions, labels)
        print(f"Accuracy: {accuracy * 100:.2f}%")  # 정확도 출력
        
        return predictions.tolist()

# Thrift 서버 실행
server = make_server(partition_thrift.Partition, PartitionHandler(), '0.0.0.0', 8080)
print("서버가 실행 중입니다...")
server.serve()
