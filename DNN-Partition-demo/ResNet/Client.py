import thriftpy2
import numpy as np
from thriftpy2.rpc import make_client
from PIL import Image
import torchvision.transforms as transforms
import os

# Partition 관련 Thrift 파일 로드
partition_thrift = thriftpy2.load("/home/user/DNN-Partition-demo/partition.thrift", module_name="partition_thrift")

def split_data(input_data, labels, chunk_size):
    """입력 데이터를 지정된 크기로 나누는 함수"""
    for i in range(0, len(input_data), chunk_size):
        yield input_data[i:i + chunk_size], labels[i:i + chunk_size]

# 서버에 연결
client = make_client(partition_thrift.Partition, '117.16.154.164', 8080)

# 이미지 전처리 (ResNet 입력에 맞게 크기 조정 및 정규화)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet은 보통 224x224 크기의 이미지를 사용
    transforms.ToTensor(),  # 이미지를 Tensor로 변환
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ResNet의 평균 및 표준편차
])

def load_and_preprocess_image(image_path):
    """이미지를 로드하고 전처리하는 함수"""
    image = Image.open(image_path).convert('RGB')  # 이미지를 RGB로 변환
    image = transform(image)  # 변환 적용
    return image

# 이미지 경로 지정
image_path = "/home/user/DNN-Partition-demo/cats.jpeg"
image = load_and_preprocess_image(image_path)

# 레이블 설정 (예시로 0으로 설정)
labels = [0]  # 이미지 하나에 대해서 레이블 하나

# 데이터 나누기 (이미지가 하나이므로 한 번만 처리)
chunk_size = 1  # 하나씩 보내는 경우
results = []

# 이미지 처리 및 서버로 전송
input_data = image.unsqueeze(0).tolist()  # 배치 차원 추가 후 리스트로 변환
result = client.partition(input_data, labels)  # 레이블과 함께 전송
results.append(result)

# 결과 처리
final_result = np.array(results)
print(f"서버에서 받은 결과: {final_result}")
