import smbus  # I2C 통신을 위한 라이브러리
import time
import cv2  # OpenCV 라이브러리
import board
import busio
import subprocess

class INA219Calculator:
    def __init__(self, i2c_address=0x40):
        self.bus = smbus.SMBus(1)  # I2C 버스 초기화
        self.address = i2c_address

    def read_shunt_voltage(self):
        # 션트 전압 읽기
        shunt_voltage_raw = self.bus.read_word_data(self.address, 0x01)
        return ((shunt_voltage_raw & 0xFF) << 8 | (shunt_voltage_raw >> 8)) * 0.01  # mV 변환

    def read_bus_voltage(self):
        # 버스 전압 읽기
        bus_voltage_raw = self.bus.read_word_data(self.address, 0x02)
        bus_voltage = ((bus_voltage_raw & 0xFF) << 8 | (bus_voltage_raw >> 8))
        return bus_voltage * 0.004  # V 변환

    def read_calibration_value(self):
        # 보정값 읽기
        calibration_raw = self.bus.read_word_data(self.address, 0x05)
        return (calibration_raw & 0xFF) << 8 | (calibration_raw >> 8)
    def calculate_current(self, shunt_voltage_mV, calibration_value):
    # 전류 계산 방식을 조정해 값을 조정해보세요
        current_lsb = 0.0001  # 필요시 조정
        current_mA = (shunt_voltage_mV / calibration_value) / current_lsb
        return current_mA

    def calculate_power(self, current_mA, bus_voltage_V):
    # 전류와 전압을 곱하여 전력 계산
        return current_mA * bus_voltage_V

def run_powerstat():
    try:
        result = subprocess.run(['sudo', 'powerstat', '-n', '1000', '-z'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            print("Error executing powerstat:", result.stderr)
            return
        print("Powerstat Output:")
        print(result.stdout)
    except Exception as e:
        print("An error occurred in powerstat: {}".format(e))

def run_powertop():
    try:
        result = subprocess.run(['sudo', 'powertop', '--html=/tmp/powertop.html'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            print("Error executing powertop:", result.stderr)
            return
        with open('/tmp/powertop.html', 'r') as file:
            output = file.read()
        print("Powertop Output:")
        print(output)
    except Exception as e:
        print("An error occurred in powertop: {}".format(e))

if __name__ == "__main__":
    calculator = INA219Calculator()
    cap = cv2.VideoCapture(0)  # 라즈베리파이 카메라 초기화

    try:
        while True:
            shunt_voltage = calculator.read_shunt_voltage()
            bus_voltage = calculator.read_bus_voltage()
            calibration_value = calculator.read_calibration_value()

            current = calculator.calculate_current(shunt_voltage, calibration_value)
            power = calculator.calculate_power(current, bus_voltage)

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

            # 'q' 키로 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            time.sleep(1)

    except KeyboardInterrupt:
        print("측정 중지.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        run_powerstat()
        run_powertop()
