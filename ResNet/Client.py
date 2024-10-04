import requests

def send_images_for_accuracy(image_paths, labels):
    url = 'http://117.16.154.164:8080/accuracy'  # <server-ip>를 서버의 IP 주소로 변경
    files = [('images', open(path, 'rb')) for path in image_paths]
    data = {'labels': labels}
    response = requests.post(url, files=files, data=data)

    if response.status_code == 200:
        print('Accuracy:', response.json()['accuracy'])
    else:
        print('Failed to get accuracy:', response.text)

if __name__ == '__main__':
    # 이미지 파일 경로와 해당 라벨 리스트
    image_paths = ['/home/user/taemin.jpeg', '/home/user/xoals.jpg']  # 실제 이미지 경로로 변경
    labels = [0, 1]  # 실제 레이블로 변경
    send_images_for_accuracy(image_paths, labels)
