import torch
import thriftpy2
import numpy as np
from thriftpy2.rpc import make_server
import sys

sys.path.append('C:/Users/USER/Desktop/AI Offloading/yolov5')  # YOLOv5 경로 추가
from models.common import DetectMultiBackend  # YOLOv5 관련 모듈
from utils.general import non_max_suppression  # YOLOv5 관련 유틸리티 함수
from utils.torch_utils import select_device

# YOLOv5 모델 로드
model_path = "C:/Users/USER/Desktop/AI Offloading/yolov5n.pt"
device = select_device('')  # 자동으로 GPU 또는 CPU 선택
model = DetectMultiBackend(model_path, device=device)

class PartitionHandler:
    def get_backbone_model(self):
        with open(model_path, 'rb') as f:
            model_data = f.read()
        return model_data

    def run_head_inference(self, image_data, neck_output, labels):
        # 이미지 데이터를 모델에 맞게 전처리
        image_tensor = torch.tensor(np.frombuffer(image_data, dtype=np.uint8)).reshape(1, 3, 32, 32).to(device)
        image_tensor = image_tensor.float() / 255.0  # float32로 변환 및 정규화

        # 모델의 일부 레이어(예: 특정 Conv 레이어)까지 추론 수행
        def run_specific_inference(image_tensor):
            model.eval()
            with torch.no_grad():
                x = image_tensor
                특정_레이어_인덱스 = 10  # 실제 분리할 레이어 인덱스 사용
                model_layers = list(model.model.model.children())  # 모델 레이어에 접근
                for i, layer in enumerate(model_layers):
                    x = layer(x)
                    if i == 특정_레이어_인덱스:
                        break
                return x

        neck_output = run_specific_inference(image_tensor)

        # Head 부분에서 최종 추론 수행
        with torch.no_grad():
            pred = model.model.model[-1](neck_output)  # Head 부분에서 최종 추론
            pred = non_max_suppression(pred)  # NMS 적용

        # 결과 처리
        results = []
        for det in pred:
            if len(det):
                for *box, conf, cls in det:
                    results.append({
                        'box': list(map(float, box)),
                        'confidence': float(conf),
                        'class': int(cls)
                    })

        # 정확도 계산
        accuracy = calculate_accuracy(results, labels)

        return {
            'results': results,
            'accuracy': accuracy
        }

def calculate_accuracy(predictions, labels):
    correct_predictions = sum(1 for pred in predictions if pred['class'] in labels)
    accuracy = correct_predictions / len(predictions) if predictions else 0
    return accuracy

# Thrift 서버 시작
partition_thrift = thriftpy2.load("C:/Users/USER/Desktop/AI Offloading/DNN-Partition-demo/partition.thrift", module_name="partition_thrift")
server = make_server(partition_thrift.Partition, PartitionHandler(), '0.0.0.0', 8080, client_timeout=60.0)
print("서버가 실행 중입니다...")
server.serve()
