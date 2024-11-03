import time
import board
import busio
import subprocess
import smbus2 as smbus

class INA219Calculator:
    def __init__(self, i2c_address=0x40):
        self.bus = smbus.SMBus(1)  # I2C 버스 초기화
        self.address = i2c_address

    def read_shunt_voltage(self):
        # 션트 전압 읽기 (단위: mV)
        shunt_voltage_raw = self.bus.read_word_data(self.address, 0x01)
        shunt_voltage = ((shunt_voltage_raw & 0xFF) << 8 | (shunt_voltage_raw >> 8)) * 0.01
        return shunt_voltage  # mV 단위 반환

    def read_bus_voltage(self):
        # 버스 전압 읽기 (단위: V)
        bus_voltage_raw = self.bus.read_word_data(self.address, 0x02)
        bus_voltage = ((bus_voltage_raw & 0xFF) << 8 | (bus_voltage_raw >> 8)) * 0.004
        return bus_voltage  # V 단위 반환

    def read_calibration_value(self):
        # 보정 값 읽기
        calibration_raw = self.bus.read_word_data(self.address, 0x05)
        calibration_value = ((calibration_raw & 0xFF) << 8 | (calibration_raw >> 8))
        return calibration_value

    def calculate_current(self, shunt_voltage_mV, calibration_value):
        # 전류 계산을 양수로 조정
        current_lsb = 0.0001  # 필요시 조정
        current_mA = abs(shunt_voltage_mV / calibration_value) / current_lsb
        return current_mA  # mA 단위 반환

    def calculate_power(self, current_mA, bus_voltage_V):
        # 전류와 전압을 곱하여 전력 계산
        return current_mA * bus_voltage_V

# powerstat 실행 함수
def run_powerstat():
    try:
        result = subprocess.run(['sudo', 'powerstat', '-n','5', '-z', '-d', '1'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            print("Error executing powerstat:", result.stderr)
        else:
            print("Powerstat Output:\n", result.stdout)
    except Exception as e:
        print("An error occurred in powerstat:", e)

# powertop 실행 함수
def run_powertop():
    try:
        result = subprocess.run(['sudo', 'powertop', '--html=/tmp/powertop.html'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            print("Error executing powertop:", result.stderr)
        else:
            with open('/tmp/powertop.html', 'r') as file:
                print("Powertop Output:\n", file.read())
    except Exception as e:
        print("An error occurred in powertop:", e)

if __name__ == "__main__":
    ina219_calculator = INA219Calculator()

    try:
        while True:
            # 션트 전압, 버스 전압 및 보정값 읽기
            shunt_voltage = ina219_calculator.read_shunt_voltage()  # 단위: mV
            bus_voltage = ina219_calculator.read_bus_voltage()      # 단위: V
            calibration_value = ina219_calculator.read_calibration_value()

            # 전류 및 전력 계산
            current = ina219_calculator.calculate_current(shunt_voltage, calibration_value)
            power = ina219_calculator.calculate_power(current, bus_voltage)

            # 측정 결과 출력
            print(f"Bus Voltage: {bus_voltage:.2f} V")
            print(f"Shunt Voltage: {shunt_voltage:.2f} mV")
            print(f"Measured Current: {current:.2f} mA")
            print(f"Power Consumption: {power:.2f} mW")
            print("------------------------------")

            time.sleep(1)  # 측정 주기 1초

    except KeyboardInterrupt:
        print("측정이 사용자에 의해 중지되었습니다.")
    finally:
        run_powerstat()  # powerstat 실행
        run_powertop()   # powertop 실행
