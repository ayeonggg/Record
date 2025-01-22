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

# 비상 착륙 함수


async def emergency_landing(drone, location):
    print(f"-- Emergency landing at location: {location}")
    await drone.action.goto_location(location[0], location[1], location[2])
    await asyncio.sleep(5)
    await drone.action.land()

# GPS 오류 보정


def correct_gps_error(last_known_location, current_location, threshold=10):
    if calculate_distance(last_known_location[0], last_known_location[1],
                          current_location[0], current_location[1]) > threshold:
        print(f"GPS error detected. Correcting location...")
        # 두 지점 간의 중간 위치로 보정
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

# 특정 위치로 이동 명령
async def fly_to_position(drone, latitude, longitude, altitude):
    print(f"-- Flying to position: lat={latitude}, lon={longitude}, alt={altitude}")
    await drone.action.goto_location(latitude, longitude, altitude)
    await asyncio.sleep(5)  # 도착을 기다리는 시간
# 바람 영향을 설정
async def set_wind(x, y, z):
    # ROS 마스터가 실행 중인지 확인
    try:
        ros_master = rosgraph.Master('/rostopic')
        ros_master.getPid()  # 마스터가 실행 중인지 확인
    except rosgraph.MasterError:
        print("ROS Master is not running. Please start roscore.")
        return

    # ROS 노드 초기화
    if not rospy.core.is_initialized():
        rospy.init_node('set_gazebo_wind', anonymous=True)

    pub = rospy.Publisher('/gazebo/set_wind', Vector3, queue_size=10)
    wind = Vector3(x, y, z)
    pub.publish(wind)
    print(f"Set wind to x: {x}, y: {y}, z: {z}")

# 바람으로 인한 기울기 변화 조정


async def adjust_for_wind(drone):
    print("-- Adjusting for wind disturbance")
    async for attitude in drone.telemetry.attitude_euler():
        roll = abs(attitude.roll_deg)
        pitch = abs(attitude.pitch_deg)

        if roll > 30 or pitch > 30:  # 30도 초과 시 조정 (라디안 값이 아님)
            print(f"Wind detected! Roll: {roll:.2f}, Pitch: {pitch:.2f}")
            # Roll, Pitch 초기화
            await drone.action.set_attitude(0.0, 0.0, 0.0, 0.5)
            await asyncio.sleep(5)
            print("Wind disturbance cleared.")
            return True  # 바람 영향 조정 완료
        return False

# 이벤트 루프


async def event_loop(drone, state_queue):
    try: 
	    # 사용자 입력 읽기
	    print("\nPress state put and Enter to update emergency states.")
	    updates={}

	    a = await ainput()
	    if a.strip().lower() == 'state':  # 'state' 입력을 확인하여 루프를 계속함
		# 상태 초기화
		print("\nUpdate emergency states:")
		updates = {}

	    # 상태별 사용자 입력 받기
		for key in emergency_states.keys():
		    value = await ainput(f"{key} (True/False): ")
		    while value not in ["True", "False"]:
		        print("Invalid input! Please enter 'True' or 'False'.")
		        value = await ainput(f"{key} (True/False): ")
		updates[key] = value == "True"  # 입력값을 Boolean으로 변환
	 # wind_detected가 True인 경우 x, y, z 입력받기
	    if updates.get("wind_detected"):
		print("Please input x, y, z wind speeds: ")
		try:
		    x = float(await ainput("x: "))  # 비동기 입력
		    y = float(await ainput("y: "))
		    z = float(await ainput("z: "))
		    await set_wind(x, y, z)  # 바람 설정 함수 호출

		    # 바람 상태 조정
		    wind_cleared = await adjust_for_wind(drone)
		    if wind_cleared:
		        updates["wind_detected"] = False
		        print("Resuming waypoint navigation...")
		except ValueError:
		    print("Invalid wind speed values! Please enter numeric values.")

		# 큐에 업데이트된 상태 추가
		await state_queue.put(updates)

		# 비상 상태 업데이트
		while not state_queue.empty():
		    key, value = await state_queue.get()
		    emergency_states[key] = value

		# 비상 상태 처리
		async for position in drone.telemetry.position():
		    current_location = (
		        position.latitude_deg,
		        position.longitude_deg,
		        position.absolute_altitude_m,
		    )

		    # GPS 오류 확인 및 보정
		    if emergency_states["gps error"]:
		        print(f"GPS error emerge")
		        if last_known_location is not None:
		            current_location = correct_gps_error(
		                last_known_location, current_location)
		            if current_location != last_known_location:
		                gps_error_count += 1
		                if gps_error_count >= 3:
		                    print(
		                        "GPS error detected 3 times, returning to home location.")
		                    await drone.action.return_to_launch()
		                    return True
		        return True

		    last_known_location = current_location

		    if emergency_states["engine_failure_detected"]:
		        print(
		            f"Engine failure detected! Current location: {current_location}")
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
async def fly_monitoring(drone, state_queue):
    print("-- Monitoring for emergencies and navigating through waypoints")
    for station in charging_stations:
        print(f"-- Flying to next destination: {station}")

        # 비상 상황 확인
        emergency_occurred = await event_loop(drone, state_queue)
        if emergency_occurred:
            print("Emergency handled. Continuing navigation.")
            continue

        # 다음 목적지로 이동
        # station은 (latitude, longitude, altitude)로 정의되어 있으므로 yaw_deg 추가
        yaw_deg = 0.0  # 드론의 방향을 0도로 설정 (북쪽을 바라봄)
        await drone.action.goto_location(station[0], station[1], station[2], yaw_deg)
        print(f"drone going to, {station[0]}, {station[1]}, {station[2]}")

        # 목적지 도착 여부 확인
        while True:
            async for position in drone.telemetry.position():
                distance = calculate_distance(
                    position.latitude_deg,
                    position.longitude_deg,
                    station[0],
                    station[1],
                )
                if distance < 5:  # 도착 판정을 위한 거리 임계값
                    print(f"-- Arrived at charging station: {station}")
                    break  # 도착하면 루프 종료
            else:
                await asyncio.sleep(1)
                continue  # 도착하지 않았으면 다시 확인
            break  # 도착 확인 후 다음 스테이션으로 이동

# 메인 실행
async def main():
    # 비상 상태 초기값
    emergency_states = {
        "wind_detected": False,
        "low_battery_detected": False,
        "engine_failure_detected": False,
        "communication_lost": False,
        "gps error": False,
    }
    # 사용자 입력 큐 생성
    state_queue = asyncio.Queue()
    drone = System()
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
    # 오프보드 모드 활성화
    
    if not await set_offboard_mode(drone):
        print("Failed to start offboard mode. Exiting...")
        return
    # 경로 rotation
    # 경로 rotation
    await drone.action.goto_location(35.1545288, 128.0933887, 30, 0)
    print("Drone heading to 35.1545288, 128.0933887, 30")

    while True:
        # 현재 위치를 가져옴
        position = await drone.telemetry.position()
        print(f"Current position: {position.latitude_deg}, {position.longitude_deg}")

        # 첫 번째 위치 도달 확인
        if position.latitude_deg == 35.1545288 and position.longitude_deg == 128.0933887:
            await drone.action.goto_location(35.153456, 128.09674, 30, 0)
            print("Drone heading to 35.153456, 128.09674, 30")

        # 두 번째 위치 도달 확인
        if position.latitude_deg == 35.153456 and position.longitude_deg == 128.09674:
            await drone.action.goto_location(35.152746, 128.09967, 30, 0)
            print("Drone heading to 35.152746, 128.09967, 30")

        # 세 번째 위치 도달 확인
        if position.latitude_deg == 35.152746 and position.longitude_deg == 128.09967:
            await drone.action.goto_location(35.151162, 128.10165, 30, 0)
            print("Drone heading to 35.151162, 128.10165, 30")

        # 네 번째 위치 도달 확인
        if position.latitude_deg == 35.151162 and position.longitude_deg == 128.10165:
            await drone.action.goto_location(35.152790, 128.10270, 30, 0)
            print("Drone heading to 35.152790, 128.10270, 30")

        # 다섯 번째 위치 도달 확인
        if position.latitude_deg == 35.152790 and position.longitude_deg == 128.10270:
            await drone.action.goto_location(35.154662, 128.10229, 30, 0)
            print("Drone heading to 35.154662, 128.10229, 30")

        # 최종 목적지로 이동
        final_destination = (35.1545288, 128.0933887, 30)
        print(f"-- Navigating to final destination: {final_destination}")
        await drone.action.goto_location(final_destination[0], final_destination[1], final_destination[2], 0)

        # 최종 도달 시 루프 종료
        if (position.latitude_deg == final_destination[0] and 
            position.longitude_deg == final_destination[1]):
            print("Final destination reached!")
            break

    print("-- Final landing initiated")
    await drone.action.land()
    await asyncio.sleep(10)
    # 연결 종료
    await drone.close()
if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    asyncio.run(main())
