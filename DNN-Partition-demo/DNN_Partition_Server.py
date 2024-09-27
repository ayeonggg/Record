import os
import torch
import thriftpy2 as thriftpy
from thriftpy2.rpc import make_server
import numpy as np 

# Partition.thrift 파일 로드
partition_thrift = thriftpy.load('C:/Users/USER/Desktop/AI Offloading/DNN-Partition-demo/partition.thrift', module_name='partition_thrift')

# Dispatcher 클래스 정의
class Dispatcher:
    def partition(self, file, ep, pp, cORs):
        try:
            # 파일 처리 로직
            for filename, content in file.items():
                file_path = f'C:/Users/USER/Desktop/AI Offloading/recv_intermediate.npy'
                with open(file_path, 'wb') as f:
                    f.write(content)
                print(f"Received and saved: {file_path}")

            # 데이터 로드 및 확인 (중간 파일 처리)
            intermediate_data = np.load(file_path)
            print(f'Loaded data shape: {intermediate_data.shape}')

            # torch tensor로 변환하여 AI 모델 처리
            tensor_data = torch.tensor(intermediate_data, dtype=torch.float32)
            print(f"Tensor data shape: {tensor_data.shape}")

            # 임의의 모델 예측 또는 처리
            # 여기에 실제 모델 inference 로직이 들어감 (가상의 처리)
            result = tensor_data.mean().item()
            print(f"Processing result: {result}")

            return f'Successfully processed file: {filename} | Result: {result}'

        except Exception as e:
            print(f"Error during file processing: {e}")
            return f"Failed to process file: {str(e)}"

# Thrift 서버 시작
def server_start():
    server = make_server(partition_thrift.Partition, Dispatcher(), '0.0.0.0', 8080)
    print("Server started on port 8080...")
    server.serve()

if __name__ == '__main__':
    server_start()
