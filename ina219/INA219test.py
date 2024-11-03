# -*- coding: utf-8 -*-
import time
import smbus2 as smbus

class INA219Calculator:
    def __init__(self, i2c_address=0x40):
        self.bus = smbus.SMBus(1)  
        self.address = i2c_address

    def read_shunt_voltage(self):
        shunt_voltage_raw = self.bus.read_word_data(self.address, 0x01)
        return ((shunt_voltage_raw & 0xFF) << 8 | (shunt_voltage_raw >> 8)) * 0.01  # mV 쨘짱횊짱

    def read_bus_voltage(self):
        bus_voltage_raw = self.bus.read_word_data(self.address, 0x02)
        bus_voltage = ((bus_voltage_raw & 0xFF) << 8 | (bus_voltage_raw >> 8))
        return bus_voltage * 0.004  
    def read_calibration_value(self):
        calibration_raw = self.bus.read_word_data(self.address, 0x05)
        return (calibration_raw & 0xFF) << 8 | (calibration_raw >> 8)

    def calculate_current(self, shunt_voltage_mV, calibration_value):
        current_lsb = 0.0001  
        if calibration_value == 0:
            print("Calibration value is 0, using default calibration.")
            calibration_value = 1  
        current_mA = (shunt_voltage_mV / calibration_value) / current_lsb
        return current_mA

    def calculate_power(self, current_mA, bus_voltage_V):
        return current_mA * bus_voltage_V

def main():
    # Initialize INA219
    ina219_calculator = INA219Calculator()

    try:
        while True:
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

            time.sleep(1)  

    except KeyboardInterrupt:
        print("stop.")
    except Exception as e:
        print(f"error {e}")

if __name__ == "__main__":
    main()
