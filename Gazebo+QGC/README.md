# 리눅스 터미널 실행 명령어

cd /home/ayeong/catkin_ws/PX4-Autopilot
[gazebo start lat,long fix command]
echo "export PX4_HOME_LAT=35.1545288" >> ~/.bashrc
echo "export PX4_HOME_LON=128.0933887" >> ~/.bashrc
echo "export PX4_HOME_ALT=30" >> ~/.bashrc
cd /home/ayeong/catkin_ws/PX4-Autopilot
make px4_sitl gazebo

/home/ayeong/QgroundControl.AppImage


[lat,log choose&twice faster]
export PX4_HOME_LAT=35.1545288
export PX4_HOME_LON=128.0933887
export PX4_HOME_ALT=30
export PX_SIM_SPEED_FACTOR=2

## 초기 코드 구상
charging_station: 2차원 리스트 튜플  
                ↓ 
                
**함수 정의**
GPS 지구 모형 반영 최단거리 계산 함수 (하버사인 공식 사용)  
최단 경로 노드 계산 함수  
비상착륙 함수 
배터리 모니터 함수 
엔진 고장 함수  
통신 에러 함수  
바람 모니터 함수  
                ↓  
                
**메인 함수** `async def run()`  
이륙 명령  
엔진 태스크 생성  
통신 태스크 생성  
배터리 태스크 생성  
바람 태스크 생성  
`if` 조건문 오류 발생 시 최단 경로로 이동 명령  
착륙 명령  
                ↓ 
                
if __name__ == "__main__":
    asyncio.run(run())


## 구현 중간 상황
charging_station :2차원 리스트 튜플
                ↓ 
                
드론오류 상태 초기값=False: 딕셔너리
                ↓ 
                
  **함수 정의**
GPS 지구 모형 반영 최단거리 계산 함수 (하버사인 공식 사용)  
최단 경로 노드 계산 함수  
비상착륙 함수 
바람 영향 함수
기울기조정
통신 오류 함수
배터리 부족 함수
                ↓ 
                
async def user_input함수 정의: if조건문
                ↓ 
                
async def fly_monitering함수 정의
                ↓ 
                
이벤트 루프: drone, state_queue인자
드론 제어 및 모니터링 fly_monitering 호출
큐 생성
                ↓ 
                
**메인 함수** `async def run()`  
이륙 명령  
사용자 드론 상태 입력user_input 호출
                ↓ 
                
if __name__ == "__main__":
    asyncio.run(run())

## 구현 결과
charging_station :2차원 리스트 튜플
                ↓ 
                
**함수 정의**
GPS 지구 모형 반영 최단거리 계산 함수 (하버사인 공식 사용)  
최단 경로 노드 계산 함수  
비상착륙 함수 
GPS error함수
오프보드 모드 활성화 함수
바람 기울기 변화 조정 함수
                ↓ 
                
이벤트 루프: drone, state_queue, emergency_states인자
while문 큐 상황 확인
if gps오류 상황
if 엔진 오류 상황
if 배터리 부족 상황
if 바람 감지 상황
                ↓
async def fly_monitering함수 정의
                
                ↓      
**메인 함수** `async def run()`  
이륙 명령 
드론오류 상태 초기값=False: 딕셔너리
큐 생성
오프로드 활성화 호출
user_input태스크 생성
async def fly_monitering함수 호출
                ↓ 
                
async def user_input함수 정의: whil True+if조건문
if a=='state'
                ↓ 
 if __name__ == "__main__":
    asyncio.run(run())               
