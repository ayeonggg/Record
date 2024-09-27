import numpy as np
import torch
import thriftpy2 as thriftpy
from thriftpy2.rpc import make_client

def client_start():
    partition_thrift = thriftpy.load('/home/user/DNN-Partition-demo/partition.thrift', module_name='partition_thrift')
    client = make_client(partition_thrift.Partition, '192.168.0.3', 8080)  # 데스크탑의 IP 주소 사용
    print('Client connected to server...')

    # 분할할 데이터를 준비
    data = np.random.randn(3, 32, 32)  # 가상 데이터 (32x32 이미지 3채널)
    np.save('/home/user/DNN-Partition-demo/intermediate.npy', data)
    print("Generated data:", data)

    with open('/home/user/DNN-Partition-demo/intermediate.npy', 'rb') as f:
        content = f.read()
    print(f'Sending data size: {len(content)} bytes')

    file = {'intermediate.npy': content}
    ep = 1  # 임의의 파라미터
    pp = 2
    cORs = 'L'  # 'L' 또는 'R'로 설정

    try:
        response = client.partition(file, ep, pp, cORs)  # cORs 추가
        print(f'Server response: {response}')
    except Exception as e:
        print(f'Error while communicating with server: {e}')

if __name__ == '__main__':
    client_start()