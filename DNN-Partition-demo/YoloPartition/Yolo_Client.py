import cv2
import torch
import thriftpy2
import numpy as np
from thriftpy2.rpc import make_client
import sys

sys.path.append('/home/user/yolov5')
from models.common import DetectMultiBackend  # YOLOv5 모델 임포트
from utils.torch_utils import select_device

# Partition 관련 Thrift 파일 로드
partition_thrift = thriftpy2.load("/home/user/DNN-Partition-demo/partition.thrift", module_name="partition_thrift")

def load_model_from_server():
    model_data = client.get_model()  # 모델 데이터를 서버에서 요청
    with open("yolov5n.pt", "wb") as f:
        f.write(model_data)
    device = select_device('')  # 자동으로 GPU 또는 CPU 선택
    model = DetectMultiBackend("yolov5n.pt", device=device)  # 모델 로드
    return model

# YOLOv5 모델 로드
model = load_model_from_server()  # 서버에서 모델 로드
model.eval()  # 평가 모드로 설정

# 카메라 열기
cap = cv2.VideoCapture(0)  # 기본 카메라 열기

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 이미지 전처리
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR에서 RGB로 변환
    img = torch.from_numpy(img).permute(2, 0, 1).float()  # 텐서로 변환 후 (C, H, W) 형식으로 변환
    img = img.unsqueeze(0)  # 배치 차원 추가 (1, C, H, W)

    # 추론 수행
    with torch.no_grad():
        results = model(img)  # 추론 수행

    # 결과를 후처리하고 화면에 표시
    for result in results:
        if result is not None and len(result) > 0:  # 결과가 존재하는 경우
            for detection in result:  # 각 탐지 결과에 대해 반복
                box = detection[:4]  # 상위 4개의 요소(x1, y1, x2, y2)만 선택
                conf = detection[4]  # 신뢰도
                cls = int(detection[5].item())  # 클래스 ID를 정수로 변환

                # box를 numpy 배열로 변환
                box = box.cpu().numpy()  
                x1, y1, x2, y2 = map(int, box)  # 각 값들을 정수로 변환
                label = f'Class {cls}: {conf:.2f}'  # 레이블 생성
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # 바운딩 박스 그리기
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)  # 클래스 레이블 그리기

    cv2.imshow('YOLOv5 Detection', frame)  # 결과를 화면에 표시
    print(detection)
    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
