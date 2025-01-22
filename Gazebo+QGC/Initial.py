#!/usr/bin/env python3

import asyncio
from mavsdk import System
import math

# Global flags and constants
wind_detected = True
low_battery_detected = False
engine_failure_detected = False
communication_lost = False
BATTERY_THRESHOLD = 30.0  # Low battery threshold in percentage
BATTERY_CRITICAL_THRESHOLD = 15.0  # Critical battery threshold in percentage

# Charging station locations (latitude, longitude, altitude)
charging_stations = [
    (35.153456, 128.09674, 30),
    (35.152746, 128.09967, 30),
    (35.151162, 128.10165, 30),
    (35.152790, 128.10270, 30),
    (35.154662, 128.10229, 30),
    (35.156474, 128.09425, 30),
]

# Function to calculate distance between two GPS points
def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371000  # Earth radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    # Haversine formula
    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c  # Distance in meters

# Function to find nearest charging station
def find_nearest_station(current_location):
    distances = [
        (calculate_distance(current_location[0], current_location[1], station[0], station[1]), station)
        for station in charging_stations
    ]
    return min(distances, key=lambda x: x[0])[1]

# Function to handle emergency landing scenarios
async def emergency_landing(drone, nearest_station=None):
    if nearest_station:
        print(f"Emergency detected! Navigating to nearest charging station at: {nearest_station}")
        await drone.action.goto_location(nearest_station[0], nearest_station[1], nearest_station[2], 0)
        await asyncio.sleep(50)  # Wait for navigation to station
    print("-- Landing initiated.")
    await drone.action.land()
    await asyncio.sleep(10)
    print("-- Emergency landing complete.")
    return

# Function to monitor battery status
async def monitor_battery(drone):
    global low_battery_detected
    async for battery in drone.telemetry.battery():
        if battery.remaining_percent < BATTERY_CRITICAL_THRESHOLD / 100.0:
            print(f"Critical battery detected! Battery level: {battery.remaining_percent * 100:.1f}%")
            low_battery_detected = True
            break
        elif battery.remaining_percent < BATTERY_THRESHOLD / 100.0:
            print(f"Low battery detected! Battery level: {battery.remaining_percent * 100:.1f}%")
            low_battery_detected = True

# Function to detect engine failure
async def monitor_engine_status(drone):
    global engine_failure_detected
    async for status in drone.telemetry.status_text():
        if "engine_failure" in status.text:  # Assuming MAVSDK reports engine failure
            print("Engine failure detected!")
            engine_failure_detected = True
            break

# Function to monitor communication status
async def monitor_communication(drone):
    global communication_lost
    try:
        async for health in drone.telemetry.health():
            if not health.is_global_position_ok:  # Example condition
                print("Communication lost detected!")
                communication_lost = True
                break
    except Exception as e:
        print(f"Communication check failed: {e}")
        communication_lost = True

# Function to get current position
async def get_current_position(drone):
    async for position in drone.telemetry.position():
        return position

# Function to detect wind disturbance
async def monitor_wind(drone):
    global wind_detected
    # Placeholder for wind detection logic
    await asyncio.sleep(5)  # Simulating wind detection logic
    wind_detected = False
    print("No wind disturbance detected.")

# Main function
async def run():
    global wind_detected, low_battery_detected, engine_failure_detected, communication_lost

    drone = System()
    await drone.connect(system_address="udp://:14540")  # Gazebo의 MAVLink 연결 주소를 확인

    print("Waiting for drone to connect...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("Drone discovered!")
            break

    print("Waiting for global position estimate...")
    async for health in drone.telemetry.health():
        if health.is_global_position_ok and health.is_home_position_ok:
            print("-- Global position state is good enough for flying.")
            break

    print("-- Arming")
    await drone.action.arm()
    await asyncio.sleep(3)

    print("-- Taking off")
    await drone.action.set_takeoff_altitude(10.0)
    await drone.action.takeoff()
    await asyncio.sleep(10)

    # Start monitoring tasks
    asyncio.create_task(monitor_engine_status(drone))
    asyncio.create_task(monitor_communication(drone))
    asyncio.create_task(monitor_battery(drone))
    asyncio.create_task(monitor_wind(drone))

    print("-- Monitoring for emergencies and navigating through waypoints")
    for station in charging_stations:
        # Check for emergency conditions
        if wind_detected or low_battery_detected or engine_failure_detected or communication_lost:
            current_location = await drone.telemetry.position()
            nearest_station = find_nearest_station((current_location.latitude_deg, current_location.longitude_deg))
            await emergency_landing(drone, nearest_station)
            return

        print(f"-- Navigating to station: {station}")
        await drone.action.goto_location(station[0], station[1], station[2], 0)
        await asyncio.sleep(30)

    # Final destination
    final_destination = (35.1545288, 128.0933887, 10)
    print(f"-- Navigating to final destination: {final_destination}")
    await drone.action.goto_location(final_destination[0], final_destination[1], final_destination[2], 0)
    await asyncio.sleep(50)

    print("-- Final landing initiated")
    await drone.action.land()
    await asyncio.sleep(10)

if __name__ == "__main__":
    asyncio.run(run())
