import asyncio
from mavsdk import System
from mavsdk.offboard import OffboardError, VelocityNedYaw
import math
from aioconsole import ainput

# 충전소 위치
charging_stations = [
    (35.1545288, 128.0933887, 30),
    (35.153456, 128.09674, 30),
    (35.152746, 128.09967, 30),
    (35.151162, 128.10165, 30),
    (35.152790, 128.10270, 30),
    (35.154662, 128.10229, 30),
    (35.156474, 128.09425, 30),
]

# 거리 계산 함수
def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371  # 지구 반지름 (km)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1)) *
        math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c * 1000  # 거리 (m)

# 가장 가까운 충전소 검색
def find_nearest_station(current_location):
    nearest_station = None
    shortest_distance = float('inf')

    for station in charging_stations:
        distance = calculate_distance(
            current_location[0], current_location[1], station[0], station[1]
        )
        if distance < shortest_distance:
            shortest_distance = distance
            nearest_station = station
    return nearest_station

# 비상 착륙 함수
async def emergency_landing(drone, location):
    print(f"-- Emergency landing at location: {location}")
    await drone.action.goto_location(location[0], location[1], location[2], 0.0)
    await asyncio.sleep(5)
    await drone.action.land()

# GPS 오류 보정
def correct_gps_error(last_known_location, current_location, threshold=10):
    if calculate_distance(last_known_location[0], last_known_location[1],
                          current_location[0], current_location[1]) > threshold:
        print(f"GPS error detected. Correcting location...")
        corrected_location = (
            (last_known_location[0] + current_location[0]) / 2,
            (last_known_location[1] + current_location[1]) / 2,
            current_location[2],
        )
        return corrected_location
    return current_location

# 오프보드 모드 활성화 함수
async def set_offboard_mode(drone):
    print("-- Setting offboard mode")
    try:
        await drone.offboard.set_velocity_ned(VelocityNedYaw(0, 0, 0, 0))
        await drone.offboard.start()
        print("-- Offboard mode activated")
    except OffboardError as e:
        print(f"Failed to start offboard mode: {e}")
        await drone.action.land()
        return False
    return True

# 바람으로 인한 기울기 변화 조정
async def adjust_for_wind(drone):
    print("-- Adjusting for wind disturbance")
    async for attitude in drone.telemetry.attitude_euler():
        roll = abs(attitude.roll_deg)
        pitch = abs(attitude.pitch_deg)

        if roll > 30 or pitch > 30:
            print(f"Wind detected! Roll: {roll:.2f}, Pitch: {pitch:.2f}")
            await drone.action.set_attitude(0.0, 0.0, 0.0, 0.5)
            await asyncio.sleep(5)
            print("Wind disturbance cleared.")
            return True
        return False

# 이벤트 루프
async def event_loop(drone, state_queue, emergency_states):
    last_known_location = None
    gps_error_count = 0

    while True:
        if not state_queue.empty():
            updates = await state_queue.get()
            for key, value in updates.items():
                emergency_states[key] = value

        async for position in drone.telemetry.position():
            current_location = (
                position.latitude_deg,
                position.longitude_deg,
                position.absolute_altitude_m,
            )

            if emergency_states["gps error"]:
                if last_known_location is not None:
                    current_location = correct_gps_error(
                        last_known_location, current_location)
                    if current_location != last_known_location:
                        gps_error_count += 1
                        if gps_error_count >= 3:
                            print("GPS error detected 3 times, returning to home location.")
                            await drone.action.return_to_launch()
                            return True
                return True

            last_known_location = current_location

            if emergency_states["engine_failure_detected"]:
                print(f"Engine failure detected! Current location: {current_location}")
                await emergency_landing(drone, current_location)
                return True

            elif emergency_states["low_battery_detected"] or emergency_states["communication_lost"]:
                nearest_station = find_nearest_station(current_location)
                print(f"Heading to nearest station: {nearest_station}")
                await emergency_landing(drone, nearest_station)
                return True

            elif emergency_states["wind_detected"]:
                wind_cleared = await adjust_for_wind(drone)
                if wind_cleared:
                    emergency_states["wind_detected"] = False
                    print("Resuming waypoint navigation...")
                    continue

            await asyncio.sleep(5)
            return False

# 드론 제어 및 모니터링
async def fly_monitoring(drone, state_queue, emergency_states):
    print("-- Monitoring for emergencies and navigating through waypoints")
    for station in charging_stations:
        print(f"-- Flying to next destination: {station}")

        emergency_occurred = await event_loop(drone, state_queue, emergency_states)
        if emergency_occurred:
            print("Emergency handled. Continuing navigation.")
            continue

        await drone.action.goto_location(station[0], station[1], station[2], 0.0)
        print(f"Drone going to {station[0]}, {station[1]}, {station[2]}")

        while True:
            async for position in drone.telemetry.position():
                distance = calculate_distance(
                    position.latitude_deg,
                    position.longitude_deg,
                    station[0],
                    station[1],
                )
                if distance < 5:
                    print(f"-- Arrived at charging station: {station}")
                    break
            else:
                await asyncio.sleep(1)
                continue
            break

# 메인 실행
async def main():
    emergency_states = {
        "wind_detected": False,
        "low_battery_detected": False,
        "engine_failure_detected": False,
        "communication_lost": False,
        "gps error": False,
    }
    state_queue = asyncio.Queue()
    drone = System()

    try:
        await drone.connect(system_address="udp://:14540")

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

        if not await set_offboard_mode(drone):
            print("Failed to start offboard mode. Exiting...")
            return

        # 사용자 입력을 관리하는 별도의 태스크 생성
        asyncio.create_task(user_input(state_queue))

        await fly_monitoring(drone, state_queue, emergency_states)

    except KeyboardInterrupt:
        print("Ctrl+C detected! Initiating landing.")
        await drone.action.land()
        await asyncio.sleep(10)
    finally:
        await drone.close()

# 사용자 입력을 관리하는 비동기 함수
async def user_input(state_queue):
    while True:
        print("\nPress 'state' and Enter to update emergency states.")
        a = await ainput()
        if a.strip().lower() == 'state':
            updates = {}
            print("\nUpdate emergency states:")
            emergency_keys = ["wind_detected", "low_battery_detected", "engine_failure_detected", "communication_lost", "gps error"]
            for key in emergency_keys:
                value = await ainput(f"{key} (True/False): ")
                while value not in ["True", "False"]:
                    print("Invalid input! Please enter 'True' or 'False'.")
                    value = await ainput(f"{key} (True/False): ")
                updates[key] = value == "True"
            await state_queue.put(updates)

if __name__ == "__main__":
    asyncio.run(main())
