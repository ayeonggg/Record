import thriftpy2
import numpy as np
from thriftpy2.rpc import make_client

# Partition 관련 Thrift 파일 로드
partition_thrift = thriftpy2.load("/home/user/DNN-Partition-demo/partition.thrift", module_name="partition_thrift")

def split_data(input_data, labels, chunk_size):
    """입력 데이터를 지정된 크기로 나누는 함수"""
    for i in range(0, len(input_data), chunk_size):
        yield input_data[i:i + chunk_size], labels[i:i + chunk_size]

# 서버에 연결
client = make_client(partition_thrift.Partition, '117.16.154.180', 8080)

# 입력 데이터 준비
input_data = np.random.rand(1, 3, 32, 32).tolist()  # 예시 입력 (CIFAR-10 이미지 사이즈)
labels = [0]  # 예시 레이블

# 데이터 나누기
chunk_size = 1
results = []

for chunk, label_chunk in split_data(input_data, labels, chunk_size):
    result = client.partition(chunk, label_chunk)
    results.append(result)

# 결과 처리
final_result = np.array(results)
print(f"서버에서 받은 결과: {final_result}")
