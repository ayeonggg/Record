import torch
import thriftpy2
import numpy as np
from thriftpy2.rpc import make_server
from pathlib import Path
import sys
sys.path.append('/home/ayeong/yolov5')  # YOLOv5 경로 추가
from models.common import DetectMultiBackend  # YOLOv5 관련 모듈
from utils.general import non_max_suppression # YOLOv5 관련 유틸리티 함수
from utils.torch_utils import select_device

# YOLOv5 모델 로드
model_path = "/home/ayeong/yolov5n.pt"
device = select_device('')  # 자동으로 GPU 또는 CPU 선택

# YOLOv5 모델 초기화
model = DetectMultiBackend(model_path, device=device)  
stride, names, pt = model.stride, model.names, model.pt

# 추론 함수 (YOLOv5)
def run_inference(data):
    data = torch.tensor(data).to(device)
    model.eval()
    
    # YOLOv5 추론
    with torch.no_grad():
        pred = model(data)  # 예측 수행
        pred = non_max_suppression(pred)  # NMS 적용
    
    # 추론 결과 처리
    results = []
    for det in pred:
        if len(det):
            for *box, conf, cls in det:
                results.append({
                    'box': box,  # bounding box 좌표
                    'confidence': conf.item(),  # confidence score
                    'class': int(cls)  # 예측된 class
                })
    return results

# Thrift 서버 핸들러 클래스 정의
class PartitionHandler:
    def partition(self, inputs, labels):
        print(f"Received input: {inputs}")
        predictions = run_inference(inputs)
        
        # YOLOv5는 bounding box 기반이므로 일반적인 '정확도' 개념이 다름
        print(f"Predictions: {predictions}")  # 예측 결과 출력
        
        return predictions  # 결과 반환

    def get_model(self):
        # 모델 파일을 바이너리로 읽어 반환
        model_path = "/home/ayeong/yolov5n.pt"
        with open(model_path, 'rb') as f:
            model_data = f.read()
        return model_data
        
# Thrift 서버 실행
partition_thrift = thriftpy2.load("/home/ayeong/Record/DNN-Partition-demo/partition.thrift", module_name="partition_thrift")
server = make_server(partition_thrift.Partition, PartitionHandler(), '0.0.0.0', 8080)
print("서버가 실행 중입니다...")
server.serve()
