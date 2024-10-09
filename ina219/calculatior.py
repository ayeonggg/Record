import smbus  # Library for I2C communication
import time

class INA219Calculator:
    def __init__(self, i2c_address=0x40):
        self.bus = smbus.SMBus(1)  # Initialize I2C bus
        self.address = i2c_address

    def read_shunt_voltage(self):
        # Read shunt voltage
        shunt_voltage_raw = self.bus.read_word_data(self.address, 0x01)
        # Bit inversion and convert to mV
        return (shunt_voltage_raw & 0xFF) << 8 | (shunt_voltage_raw >> 8)

    def read_bus_voltage(self):
        # Read bus voltage
        bus_voltage_raw = self.bus.read_word_data(self.address, 0x02)
        # Bit inversion and convert to V
        bus_voltage = (bus_voltage_raw & 0xFF) << 8 | (bus_voltage_raw >> 8)
        return bus_voltage * 0.004  # Convert to V

    def read_calibration_value(self):
        # Read calibration value
        calibration_raw = self.bus.read_word_data(self.address, 0x05)
        return (calibration_raw & 0xFF) << 8 | (calibration_raw >> 8)

    def calculate_current(self, shunt_voltage_mV, calibration_value):
        current_lsb = (shunt_voltage_mV * 20e-3) / calibration_value
        return current_lsb  # Current in mA

    def calculate_power(self, current_mA, bus_voltage_V):
        return (current_mA * bus_voltage_V) / 5000  # Power in mW

# Example usage
if __name__ == "__main__":
    calculator = INA219Calculator()

    while True:
        shunt_voltage = calculator.read_shunt_voltage()  # Read in mV
        bus_voltage = calculator.read_bus_voltage()  # Read in V
        calibration_value = calculator.read_calibration_value()  # Read calibration value

        # Calculate current and power
        current = calculator.calculate_current(shunt_voltage, calibration_value)
        power = calculator.calculate_power(current, bus_voltage)

        # Output results
        print(f"Measured Current: {current:.2f} mA")
        print(f"Power Consumption: {power:.2f} mW")

        time.sleep(1)  #efresh data every second
