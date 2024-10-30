import time
import cv2
import torch
import datetime
import os
import smbus  # SMBus 모듈 가져오기
import subprocess

class INA219Calculator:
    def __init__(self, i2c_address=0x40):
        self.bus = smbus.SMBus(1)  # I2C 버스 초기화
        self.address = i2c_address

    def read_shunt_voltage(self):
        shunt_voltage_raw = self.bus.read_word_data(self.address, 0x01)
        return ((shunt_voltage_raw & 0xFF) << 8 | (shunt_voltage_raw >> 8)) * 0.01  # mV 변환

    def read_bus_voltage(self):
        bus_voltage_raw = self.bus.read_word_data(self.address, 0x02)
        bus_voltage = ((bus_voltage_raw & 0xFF) << 8 | (bus_voltage_raw >> 8))
        return bus_voltage * 0.004  # V 변환

    def read_calibration_value(self):
        calibration_raw = self.bus.read_word_data(self.address, 0x05)
        return (calibration_raw & 0xFF) << 8 | (calibration_raw >> 8)
   
    def calculate_current(self, shunt_voltage_mV, calibration_value):
        current_lsb = 0.0001  # 필요시 조정
        if calibration_value == 0:
            print("Calibration value is 0, using default calibration.")
            calibration_value = 1  # 기본값 설정
        current_mA = (shunt_voltage_mV / calibration_value) / current_lsb
        return current_mA

    def calculate_power(self, current_mA, bus_voltage_V):
        return current_mA * bus_voltage_V

def save_image(image):
    now = datetime.datetime.now()
    filename = f'/home/user/yolov5/output/{now.strftime("%Y%m%d_%H%M%S")}.jpg'
    cv2.imwrite(filename, image)
    print(f"Image saved as {filename}")

def main():
    # Initialize INA219
    ina219_calculator = INA219Calculator()

    # Load YOLOv5 model
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='/home/user/yolov5/yolov5s.pt', force_reload=True)

    # 비디오 캡처 초기화
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    try:
        while True:
            # 전력 데이터 측정
            shunt_voltage = ina219_calculator.read_shunt_voltage()
            bus_voltage = ina219_calculator.read_bus_voltage()
            calibration_value = ina219_calculator.read_calibration_value()

            current = ina219_calculator.calculate_current(shunt_voltage, calibration_value)
            power = ina219_calculator.calculate_power(current, bus_voltage)

            print(f"Bus Voltage: {bus_voltage:.2f} V")
            print(f"Shunt Voltage: {shunt_voltage:.2f} mV")
            print(f"Measured Current: {current:.2f} mA")
            print(f"Power Consumption: {power:.2f} mW")
            print("------------------------------")

            # 비디오 프레임 읽기
            ret, frame = cap.read()
            if not ret:
                print("카메라에서 프레임을 읽지 못했습니다.")
                break

            # 영상에 측정값 표시
            cv2.putText(frame, f"Bus Voltage: {bus_voltage:.2f} V", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Current: {current:.2f} mA", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Power: {power:.2f} mW", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # 화면에 표시
            cv2.imshow("Power Consumption Analysis", frame)

            # 이미지 저장 (옵션)
            save_image(frame)  # 매 프레임마다 저장 원하지 않으면 이 줄을 주석 처리

            # 'q' 키로 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            time.sleep(1)

    except KeyboardInterrupt:
        print("측정 중지.")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
