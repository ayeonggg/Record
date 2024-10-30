import time
import threading
import cv2
import torch
from datetime import datetime
from pathlib import Path
import smbus  # I2C 통신 라이브러리

class INA219Calculator:
    def __init__(self, i2c_address=0x40):
        self.bus = smbus.SMBus(1)  # I2C 버스 초기화
        self.address = i2c_address
        self.current_mA = 0  # 초기 전류값

    def read_shunt_voltage(self):
        shunt_voltage_raw = self.bus.read_word_data(self.address, 0x01)
        shunt_voltage = ((shunt_voltage_raw & 0xFF) << 8 | (shunt_voltage_raw >> 8)) * 0.01
        return shunt_voltage

    def read_bus_voltage(self):
        bus_voltage_raw = self.bus.read_word_data(self.address, 0x02)
        bus_voltage = ((bus_voltage_raw & 0xFF) << 8 | (bus_voltage_raw >> 8)) * 0.004
        return bus_voltage

    def read_calibration_value(self):
        calibration_raw = self.bus.read_word_data(self.address, 0x05)
        calibration_value = ((calibration_raw & 0xFF) << 8 | (calibration_raw >> 8))
        return calibration_value

    def calculate_current(self, shunt_voltage_mV, calibration_value):
        current_lsb = 0.0001  # 필요시 조정
        self.current_mA = abs(shunt_voltage_mV / calibration_value) / current_lsb
        return self.current_mA

    def calculate_power(self, current_mA, bus_voltage_V):
        return current_mA * bus_voltage_V

    def update_power_reading(self):
        while True:
            shunt_voltage = self.read_shunt_voltage()
            bus_voltage = self.read_bus_voltage()
            calibration_value = self.read_calibration_value()
            current = self.calculate_current(shunt_voltage, calibration_value)
            power = self.calculate_power(current, bus_voltage)

            print(f"Bus Voltage: {bus_voltage:.2f} V")
            print(f"Shunt Voltage: {shunt_voltage:.2f} mV")
            print(f"Measured Current: {current:.2f} mA")
            print(f"Power Consumption: {power:.2f} mW")
            print("------------------------------")
            time.sleep(1)  # 1초마다 측정

class YOLODetector:
    def __init__(self, model_path='/home/user/yolov5/yolov5s.pt', save_dir='/home/user/yolov5/output/'): 
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def detect_and_save(self, power_calculator):
        cap = cv2.VideoCapture(0)  # 웹캠 또는 비디오 입력
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = self.model(frame)  # YOLOv5 추론
            annotated_frame = frame.copy()  # 원본 프레임 복사본 생성
            results.render(annotated_frame)  # 바운딩 박스 결과를 복사본에 렌더링

            # 전력 측정 값 텍스트 표시
            power_text = f"Power: {power_calculator.current_mA * power_calculator.bus_voltage:.2f} mW"
            cv2.putText(annotated_frame, power_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # 프레임 저장
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            cv2.imwrite(str(self.save_dir / f"{timestamp}.jpg"), annotated_frame)

            # 화면에 결과 출력
            cv2.imshow("YOLO Detection", annotated_frame)

            # 'q'를 누르면 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            time.sleep(1)  # 초당 1장씩 저장

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    ina219_calculator = INA219Calculator()
    yolo_detector = YOLODetector()

    # 전력 측정 및 YOLOv5 실행을 각각의 스레드에서 실행
    power_thread = threading.Thread(target=ina219_calculator.update_power_reading)
    detection_thread = threading.Thread(target=yolo_detector.detect_and_save, args=(ina219_calculator,))

    power_thread.start()
    detection_thread.start()

    power_thread.join()
    detection_thread.join()
