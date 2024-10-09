import time
import board
import busio
import subprocess
import smbus  # Library for I2C communication

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
        return current_lsb  # Return current in mA

    def calculate_power(self, current_mA, bus_voltage_V):
        return (current_mA * bus_voltage_V) 


# Initialize I2C and INA219
i2c_bus = busio.I2C(board.SCL, board.SDA)
ina219_calculator = INA219Calculator()

def run_powerstat():
    try:
        result = subprocess.run(['sudo', 'powerstat', '-n', '1000', '-z'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        if result.returncode != 0:
            print("Error executing powerstat:", result.stderr)
            return
        
        output = result.stdout
        
        if not output:
            print("No output from powerstat.")
            return
        
        print("Powerstat Output:")
        print(output)
        
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
        
        if not output:
            print("No output from powertop.")
            return

        print("Powertop Output:")
        print(output)

    except Exception as e:
        print("An error occurred in powertop: {}".format(e))

if __name__ == "__main__":
    # Measure power data continuously
    try:
        while True:
            shunt_voltage = ina219_calculator.read_shunt_voltage()  # Read shunt voltage in mV
            bus_voltage = ina219_calculator.read_bus_voltage()  # Read bus voltage in V
            calibration_value = ina219_calculator.read_calibration_value()  # Read calibration value

            # Calculate current and power
            current = ina219_calculator.calculate_current(shunt_voltage, calibration_value)
            power = ina219_calculator.calculate_power(current, bus_voltage)

            # Output results
            print(f"Bus Voltage: {bus_voltage:.2f} V")
            print(f"Shunt Voltage: {shunt_voltage:.2f} mV")
            print(f"Measured Current: {current:.2f} mA")
            print(f"Power Consumption: {power:.2f} mW")
            print("------------------------------")
            
            time.sleep(1)  # Set measurement interval to 1 second

    except KeyboardInterrupt:
        print("Measurement stopped by user.")
    finally:
        run_powerstat()  # Run powerstat
        run_powertop()   # Run powertop
