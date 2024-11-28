import thriftpy2
import numpy as np
from thriftpy2.rpc import make_client
from PIL import Image
import torchvision.transforms as transforms
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Partition 관련 Thrift 파일 로드
partition_thrift = thriftpy2.load("/home/user/DNN-Partition-demo/partition.thrift", module_name="partition_thrift")

# 서버에 연결
client = make_client(partition_thrift.Partition, '117.16.154.180', 8080)

# 이미지 전처리 (ResNet 입력에 맞게 크기 조정 및 정규화)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_and_preprocess_image(image_path):
    """이미지를 로드하고 전처리하는 함수"""
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    return image

# 이미지 경로 지정
image_path = "/home/user/DNN-Partition-demo/cats.jpeg"
image = load_and_preprocess_image(image_path)

# 레이블 설정 (예시로 0으로 설정)
labels = np.array([0])

# 이미지 처리 및 서버로 전송
input_data = image.unsqueeze(0).tolist()  # 배치 차원을 추가
result = client.partition(input_data, labels)

# 결과 처리
final_result = np.array(result)

# 서버 결과의 평균값 계산 (결과가 여러 차원일 경우 단순화)
final_result_flattened = final_result.mean(axis=(1, 2, 3)) if final_result.ndim > 1 else final_result

# 실제 값과 비교 (예: 임의로 설정된 실제 값)
actual_values = np.array([0.5])  # 결과와 크기가 일치하도록 설정

# 성능 평가 지표 계산
mse = mean_squared_error(actual_values, final_result_flattened)
rmse = np.sqrt(mse)
mae = mean_absolute_error(actual_values, final_result_flattened)

# 출력
print(f"서버에서 받은 결과(평균값): {final_result_flattened}")
print(f"MSE (Mean Squared Error): {mse:.4f}")
print(f"RMSE (Root Mean Squared Error): {rmse:.4f}")
print(f"MAE (Mean Absolute Error): {mae:.4f}")
