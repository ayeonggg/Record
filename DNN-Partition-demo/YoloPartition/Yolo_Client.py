import thriftpy2
from thriftpy2.rpc import make_client
import numpy as np
from PIL import Image

# Partition 관련 Thrift 파일 로드
partition_thrift = thriftpy2.load("/home/user/DNN-Partition-demo/partition.thrift", module_name="partition_thrift")
def main():
    # 서버에 연결
    try:
        client = make_client(partition_thrift.Partition, '192.168.0.3', 8080)
        print("서버에 연결되었습니다.")
    except Exception as e:
        print(f"서버에 연결할 수 없습니다: {e}")
        return

    # 이미지 파일을 읽어들여 바이트 배열로 변환
    image_path = "/home/user/luna.jpg"
    try:
        with Image.open(image_path) as img:
            img = img.resize((32, 32))
            image_data = np.array(img).astype(np.uint8).tobytes()
            print("이미지 데이터를 성공적으로 준비했습니다.")
    except Exception as e:
        print(f"이미지를 처리하는 중 오류 발생: {e}")
        return

    # 예시 neck_output 및 labels
    neck_output = np.zeros((1, 3, 32, 32)).flatten().tolist()
    labels = [0]

    # 서버로 데이터 전송 및 응답 받기
    try:
        response = client.run_head_inference(image_data, neck_output, labels)
        print("서버로부터 응답을 받았습니다.")
        print("Results:", response['results'])
        print("Accuracy:", response['accuracy'])
    except Exception as e:
        print(f"데이터 전송 중 오류 발생: {e}")

if __name__ == "__main__":
    main()
