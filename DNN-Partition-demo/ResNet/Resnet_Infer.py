import torch
import os
from config import PARAM_PATH
from Model_Pair import ResNetExit1Part1L, ResNetExit1Part1R, ResNetExit2Part1L, ResNetExit2Part1R, \
                      ResNetExit3Part1L, ResNetExit3Part1R, ResNetExit4Part1L, ResNetExit4Part1R

# 모델 매핑 딕셔너리 정의 (클래스 이름을 문자열로 매핑)
netMapping = {
    'NetExit1Part1': [ResNetExit1Part1L, ResNetExit1Part1R],
    'NetExit2Part1': [ResNetExit2Part1L, ResNetExit2Part1R],
    'NetExit3Part1': [ResNetExit3Part1L, ResNetExit3Part1R],
    'NetExit4Part1': [ResNetExit4Part1L, ResNetExit4Part1R]
}

def infer(image, netPair, ep, pp, cORs):
    try:
        # netPair에 해당하는 모델 클래스 이름 리스트 가져오기
        class_names = netMapping[netPair]

        # cORs에 따라 'L' 또는 'R' 모델 선택
        if cORs == 'L':
            class_name = class_names[0]  # 'L'일 경우 첫 번째 모델 클래스
        elif cORs == 'R':
            class_name = class_names[1]  # 'R'일 경우 두 번째 모델 클래스
        else:
            raise ValueError(f"Invalid cORs value: {cORs}")
        
        # 동적으로 클래스 가져오기
        net_class = getattr(Model_Pair, class_name)
        
        # 모델 인스턴스 생성
        net = net_class()

        # 모델 파일 이름을 동적으로 설정
        if ep == 3 and pp == 2 and cORs == 'L':
            params_path = '/home/user/DNN-Partition-demo/alexnet_data_out/models/NetExit3Part2_L.pth'
        elif ep == 3 and pp == 2 and cORs == 'R':
            params_path = '/home/user/DNN-Partition-demo/alexnet_data_out/models/NetExit3Part2_R.pth'
        else:
            params_path = f'C:/Users/USER/Desktop/AI Offloading/DNN-Partition-demo/alexnet_data_out/models/model_epoch_{ep}.pth'

        # 파일이 존재하는지 확인
        if not os.path.exists(params_path):
            raise FileNotFoundError(f"No such file: '{params_path}'")

        # 디바이스 설정 (CUDA 사용 가능하면 GPU로, 그렇지 않으면 CPU로 설정)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 모델 가중치 로드
        net.load_state_dict(torch.load(params_path, map_location=device), strict=False)
        net.to(device)  # 모델을 디바이스로 이동
        net.eval()

        # 입력 데이터가 Tensor가 아니면 변환
        if not isinstance(image, torch.Tensor):
            image = torch.tensor(image, dtype=torch.float32)
        
        # 배치 차원 추가
        image = image.unsqueeze(0)
        image = image.to(device)

        # 모델 추론
        with torch.no_grad():
            output = net(image)

        return output

    except KeyError as e:
        print(f"Error: Unknown netPair '{netPair}'")
        raise ValueError(f"Unknown netPair: {netPair}") from e
    except Exception as e:
        print(f"An error occurred: {e}")
        raise
