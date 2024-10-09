import time
import board
import busio
import subprocess
from adafruit_ina219 import INA219

# Initialize I2C and INA219
i2c_bus = busio.I2C(board.SCL, board.SDA)
ina219 = INA219(i2c_bus)

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
            bus_voltage = ina219.bus_voltage
            shunt_voltage = ina219.shunt_voltage
            current = ina219.current
            
            print(f"Bus Voltage: {bus_voltage} V")
            print(f"Shunt Voltage: {shunt_voltage} mV")
            print(f"Current: {current} mA")
            print("------------------------------")
            
            time.sleep(1)  # Adjust the interval as needed

    except KeyboardInterrupt:
        print("Measurement stopped by user.")
    finally:
        run_powerstat()
        run_powertop()
