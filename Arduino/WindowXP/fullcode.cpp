#include <dht.h>
#include <pm2008_i2c.h>
#include <Servo.h>

// 릴레이 핀 정의
int Relaypin = 8;

// 객체 인스턴스화
dht DHT;
PM2008_I2C pm2008_i2c;
Servo myservo;

// 상수
#define DHT21_PIN 2   // DHT 21 (AM2302) - 연결된 핀
#define DHT21_PIN2 4  // DHT 21 (AM2302) - 두 번째 센서 핀
#define LEDPIN 11     // LED 밝기 (PWM) 쓰기
#define LIGHTPIN A0   // 주변광 센서 읽기
#define RAINPIN A1    // 비 센서 아날로그 입력 핀
#define PIEZO 7       // 피에조 버저 핀
#define INTERVAL 10   // 10vvvv분 1분 간격 (10 샘플)
#define SERVO 10

// 변수
float hum;              // 습도 값 저장
float temp;             // 온도 값 저장
float hum2;             // 두 번째 센서의 습도 값 저장
float temp2;            // 두 번째 센서의 온도 값 저장
float userHumi = 50.0;  // 기본 표준 습도
float userHot = 26;     // 기본 표준 높은 온도
float userCool = 24.0;  // 기본 표준 낮은 온도
float userDust = 81;
float userLight = 400;
float reading;          // 조도 값 저장
int rainValue;          // 비 센서 값 저장
float pm25Value;        // PM2.5 값 저장
int currentIndex = 0;
unsigned long lastSampleTime = 0;
String inputCommand = "";  // 사용자 입력 저장 변수
unsigned long before = 0;  // 10분 타이머 추적
int pos = 0;

// 10분 동안 센서 값을 저장하는 배열
float humValues[INTERVAL] = { 0 };
float tempValues[INTERVAL] = { 0 };
float hum2Values[INTERVAL] = { 0 };
float temp2Values[INTERVAL] = { 0 };
float lightValues[INTERVAL] = { 0 };
float rainValues[INTERVAL] = { 0 };
float pm25Values[INTERVAL] = { 0 };

void clearSerialBuffer() {
  while (Serial.available() > 0) {
    Serial.read();  // 버퍼 지우기
  }
}

float getValidFloatInput(String prompt) {
  Serial.println(prompt);

  clearSerialBuffer();  // 버퍼 지우기

  while (true) {
    if (Serial.available() > 0) {
      float value = Serial.parseFloat();  // 실수 입력 읽기

      // 디버그 메시지
      Serial.print(F("Received value: "));
      Serial.println(value);

      if (value > 0) {  // 0보다 큰 값만 허용
        return value;
      } else {
        Serial.println(F("Invalid value. Please enter again."));
        Serial.println(prompt);  // 프롬프트 다시 출력
      }

      clearSerialBuffer();  // 입력 버퍼 지우기
    }
    delay(100);  // 입력 대기
  }
}

void setUserHumi() {
  Serial.println(F("Set the humidity: "));
  userHumi = getValidFloatInput(F("Set the humidity (float greater than 0): "));
  Serial.print(F("Set humidity: "));
  Serial.println(userHumi);
}

void setUserHot() {
  Serial.println(F("\nSet the hot temperature: "));
  userHot = getValidFloatInput(F("Set the hot temperature (float greater than 0): "));
  Serial.print(F("Set hot temperature: "));
  Serial.println(userHot);
}

void setUserCool() {
  Serial.println(F("\nSet the cool temperature: "));
  userCool = getValidFloatInput(F("Set the cool temperature (float greater than 0): "));
  Serial.print(F("Set cool temperature: "));
  Serial.println(userCool);
}

void setUserDust() {
  Serial.println(F("Set the dust: "));
  userDust = getValidFloatInput(F("Set the dust (float greater than 0): "));
  Serial.print(F("Set dust: "));
  Serial.println(userDust);
}


void setUserLight() {
  Serial.println(F("Set the light: "));
  userLight = getValidFloatInput(F("Set the light (float greater than 0): "));
  Serial.print(F("Set light: "));
  Serial.println(userLight);
}

void askToSetValues() {
  Serial.println(F(" "));
  Serial.println(F("Do you want to set the values? (y/n)"));

  while (true) {
    if (Serial.available() > 0) {
      String response = Serial.readStringUntil('\n');
      response.trim();

      // 추가로 공백 및 특수 문자 제거
      response.replace("\r", "");
      response.replace("\n", "");
      response.replace("\t", "");

      Serial.print(F("Received value: "));  // 디버그 메시지
      Serial.println(response);

      if (response.equalsIgnoreCase("y")) {
        setUserHumi();
        setUserHot();
        setUserCool();
        Serial.println(F("\n10 minute timer has been reset."));
        break;
      } else if (response.equalsIgnoreCase("n")) {
        Serial.println(F("Running with initial values."));
        break;
      } else {
        Serial.println(F("Invalid value. Please enter again."));
      }
    }
    delay(100);  // 입력 대기
  }
}

void processUserInput() {
  if (Serial.available() > 0) {
    inputCommand = Serial.readStringUntil('\n');
    inputCommand.trim();  // 여분의 공백 제거

    // 디버그 메시지
    Serial.print(F("Received value: "));
    Serial.println(inputCommand);

    if (inputCommand.equalsIgnoreCase("설정")) {
      delay(1000);
      // 초기 상태로 복귀 설정
      currentIndex = 0;
      lastSampleTime = 0;
      before = millis(); // 타이머 재설정을 위해 현재 시간 저장
      clearSerialBuffer(); // 시리얼 버퍼 지우기
      askToSetValues();
    } else if (inputCommand == "2") {
      digitalWrite(Relaypin, HIGH);
    } else if (inputCommand == "3") {
      digitalWrite(Relaypin, LOW);
    }
  }
}

float calculateDiscomfortIndex(float temp, float hum) {
  return (9.0 / 5.0) * temp - 0.55 * (1.0 - (hum / 100.0)) * ((9.0 / 5.0) * temp - 26.0) + 32.0;
}

void printSensorReadings() {
  Serial.print(F("Humidity 1: "));
  Serial.print(hum, 1);
  Serial.println(F("%"));

  Serial.print(F("Temperature 1: "));
  Serial.print(temp, 1);
  Serial.println(F("°C"));

  Serial.print(F("Humidity 2: "));
  Serial.print(hum2, 1);
  Serial.println(F("%"));

  Serial.print(F("Temperature 2: "));
  Serial.print(temp2, 1);
  Serial.println(F("°C"));

  Serial.print(F("Light Level: "));
  Serial.println(reading, 2);

  if (reading >= 400) {
    Serial.println(F("Very Light"));
  } else if (reading >= 200 && reading < 400) {
    Serial.println(F("Normal Light"));
  } else if (reading < 200) {
    Serial.println(F("Very Dark"));
  }

  Serial.print(F("Rain Level: "));
  Serial.println(rainValue);

  if (rainValue < 500) {
    Serial.println(F("Heavy Rain"));
  } else if (rainValue >= 500 && rainValue < 900) {
    Serial.println(F("Moderate Rain"));
  } else {
    Serial.println(F("No Rain"));
  }

  Serial.print(F("PM2.5 Level: "));
  Serial.println(pm25Value, 2);

  if (pm25Value > 0 && pm25Value <= 30) {
    Serial.println(F("Fine dust concentration is very good"));
  } else if (pm25Value > 30 && pm25Value <= 80) {
    Serial.println(F("Fine dust concentration is normal"));
  } else if (pm25Value > 80 && pm25Value <= 150) {
    Serial.println(F("Fine dust concentration is bad"));
  } else if (pm25Value > 150) {
    Serial.println(F("Fine dust concentration is very bad"));
  }

  // 불쾌지수 계산
  float discomfortIndex1 = calculateDiscomfortIndex(temp, hum);
  float discomfortIndex2 = calculateDiscomfortIndex(temp2, hum2);

  Serial.print(F("Discomfort Index 1: "));
  Serial.println(discomfortIndex1, 2);

  if (discomfortIndex1 >= 80) {
    Serial.println(F("The discomfort index is very high!"));
  } else if (discomfortIndex1 >= 75) {
    Serial.println(F("The discomfort index is high"));
  } else if (discomfortIndex1 >= 68) {
    Serial.println(F("It's comfortable outside"));
  } else {
    Serial.println(F("It's very comfortable outside!"));
  }

  Serial.print(F("Discomfort Index 2: "));
  Serial.println(discomfortIndex2, 2);

  if (discomfortIndex2 >= 80) {
    Serial.println(F("The discomfort index is very high!\n"));
  } else if (discomfortIndex2 >= 75) {
    Serial.println(F("The discomfort index is high\n"));
  } else if (discomfortIndex2 >= 68) {
    Serial.println(F("The interior is comfort\n"));
  } else {
    Serial.println(F("The interior is very comfort!\n"));
  }
}

float calculateAverage(float* arr, int length) {
  float sum = 0;
  int count = 0;
  for (int i = 0; i < length; i++) {
    if (arr[i] != 0) {  // 0이 아닌 값만 고려
      sum += arr[i];
      count++;
    }
  }
  return (count > 0) ? (sum / count) : 0;  // 0으로 나누지 않기 위해
}

void controlServoBasedOnAverage(float avgTemp1, float avgHum1, float avgTemp2, float avgHum2) {
  float discomfortIndex1 = calculateDiscomfortIndex(avgTemp1, avgHum1);
  float discomfortIndex2 = calculateDiscomfortIndex(avgTemp2, avgHum2);


  Serial.print(F("10 minute average discomfort index 1: "));
  Serial.println(discomfortIndex1, 2);


  Serial.print(F("10 minute average discomfort index 2: "));
  Serial.println(discomfortIndex2, 2);


  // 상황에 따른 서보 제어
  if (avgTemp2 > userHot || avgHum2 > userHumi || discomfortIndex1 < discomfortIndex2 || temp < temp2) {
    myservo.write(180);  // 서보를 열림 위치로 회전
    Serial.println(F("door Opened"));
  } else if (temp2 < userCool || pm25Value>userDust || (discomfortIndex1 > discomfortIndex2 || temp > temp2 || discomfortIndex1 >= 70 || temp >= 30 || hum >= 80 || rainValue > 1000 || pm25Value >= 81) {
    myservo.write(0);  // 서보를 닫힘 위치로 회전
    Serial.println(F("door Closed"));
  } else {
    myservo.write(90);  // 중립 위치
    Serial.println(F("door Neutral"));
  }


  if (reading> userLight){
    for (pos = 65; pos >= 0; pos -= 1) {  // 65도에서 0도까지 이동
      myservo.write(pos);                 // 서보를 'pos' 위치로 이동
      break;                              // 서보가 위치에 도달할 때까지 15ms 대기
    }
   
  }
}

void setup() {
  Serial.begin(9600);      // 시리얼 통신 시작
  pinMode(LIGHTPIN, INPUT);  // 조도 센서 핀을 입력으로 설정
  pinMode(LEDPIN, OUTPUT);   // LED 핀을 출력으로 설정
  pinMode(RAINPIN, INPUT);   // 비 센서 핀을 입력으로 설정
  pinMode(PIEZO, OUTPUT);    // 피에조 핀을 출력으로 설정
  pinMode(Relaypin, OUTPUT); // 릴레이 핀을 출력으로 설정
  myservo.attach(SERVO);
  setup_10avg();

  // PM2008 센서 초기화
  pm2008_i2c.begin();
  pm2008_i2c.command();
  delay(10);

  // 초기 설정
  askToSetValues();
}

void loop() {
  unsigned long now = millis();  // 현재 시간 업데이트

  // 10분 타이머 확인
  if (now - before >= 600000) {  // 10분 = 600,000밀리초
    Serial.println(F("------------10 minutes have passed.------------"));
    loop_10avg();  // 10분 후에 실행할 코드를 추가
    before = now;  // 타이머 재설정
  } else {
    printSensorReadings();
  }

  // 사용자 입력 처리
  processUserInput();

  // 첫 번째 DHT-21 센서에서 데이터 읽기
  int chk = DHT.read21(DHT21_PIN);
  hum = DHT.humidity;
  temp = DHT.temperature;
  delay(100);

  // 두 번째 DHT-21 센서에서 데이터 읽기
  chk = DHT.read21(DHT21_PIN2);
  hum2 = DHT.humidity;
  temp2 = DHT.temperature;

  // 주변광 센서에서 조도 값 읽기
  reading = analogRead(LIGHTPIN);
  float square_ratio = reading / 1023.0;
  square_ratio = pow(square_ratio, 2.0);

  // 상대적으로 LED 밝기 조절
  analogWrite(LEDPIN, 255.0 * square_ratio);

  // 비 센서 값 읽기 및 시리얼 모니터에 상태 출력
  rainValue = analogRead(RAINPIN);  // A1 핀에서 비 센서 읽기

  // PM2008 센서에서 데이터 읽기
  uint8_t ret = pm2008_i2c.read();
  if (ret == 0) {
    pm25Value = pm2008_i2c.number_of_2p5_um;
  }

  // PDLC 제어
  if (reading >= 50) {
    for (pos = 65; pos >= 0; pos -= 1) {  // 65도에서 0도까지 이동
      myservo.write(pos);                 // 서보를 'pos' 위치로 이동
      break;                              // 서보가 위치에 도달할 때까지 15ms 대기
    }
  } else {
    for (pos = 0; pos <= 65; pos += 1) {  // 0도에서 65도까지 이동
      myservo.write(pos);                 // 서보를 'pos' 위치로 이동
      break;
    }
  }

  // 1분 간격으로 배열에 값 저장
  if (now - lastSampleTime >= 60000) {  // 1분 = 60,000밀리초
    humValues[currentIndex] = hum;
    tempValues[currentIndex] = temp;
    hum2Values[currentIndex] = hum2;
    temp2Values[currentIndex] = temp2;
    lightValues[currentIndex] = reading;
    rainValues[currentIndex] = rainValue;
    pm25Values[currentIndex] = pm25Value;

    currentIndex++;
    lastSampleTime = now;

    // 인덱스가 간격을 초과하면 재설정
    if (currentIndex >= INTERVAL) {
      currentIndex = 0;
    }
  }

  delay(500);
}

void setup_10avg() {
  // 여기서는 추가 설정이 필요하지 않으며, 메인 설정에서 이미 호출됨
}

void loop_10avg() {
  Serial.println(F("Calculating averages..."));

  float avgHum = calculateAverage(humValues, INTERVAL);
  float avgTemp = calculateAverage(tempValues, INTERVAL);
  float avgHum2 = calculateAverage(hum2Values, INTERVAL);
  float avgTemp2 = calculateAverage(temp2Values, INTERVAL);
  float avgLight = calculateAverage(lightValues, INTERVAL);
  float avgRain = calculateAverage(rainValues, INTERVAL);
  float avgPM25 = calculateAverage(pm25Values, INTERVAL);

  delay(100);

  // 디버그용 값 출력
  Serial.print(F("Average Humidity (Sensor 1): "));
  Serial.println(avgHum);
  Serial.print(F("Average Temperature (Sensor 1): "));
  Serial.println(avgTemp);
  Serial.print(F("Average Humidity (Sensor 2): "));
  Serial.println(avgHum2);
  Serial.print(F("Average Temperature (Sensor 2): "));
  Serial.println(avgTemp2);
  Serial.print(F("Average Light Level: "));
  Serial.println(avgLight);
  Serial.print(F("Average Rain Level: "));
  Serial.println(avgRain);
  Serial.print(F("Average PM2.5 Level: "));
  Serial.println(avgPM25);

  delay(100);

  // 평균 값에 따라 서보 제어
  controlServoBasedOnAverage(avgTemp, avgHum, avgTemp2, avgHum2);

  // 인덱스 재설정
  currentIndex = 0;
}
