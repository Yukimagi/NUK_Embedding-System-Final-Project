#include <DHT.h>

// 定義 DHT11 的引腳和類型
#define DHTPIN 2   // DHT11 Data 腳連接到 D2
#define DHTTYPE DHT11
#define BUZZER_PIN 9  // 蜂鳴器連接的腳位

DHT dht(DHTPIN, DHTTYPE);

// 定義音符對應的頻率 (Hz)
#define NOTE_DO 262  // C4
#define NOTE_RE 294  // D4
#define NOTE_MI 330  // E4
#define NOTE_FA 349  // F4
#define NOTE_SOL 392 // G4
#define NOTE_LA 440  // A4
void playMedicationTone();
void playBloodPressureTone();
void playExerciseTone();
void playNote(int frequency, int duration);


void setup() {
  Serial.begin(9600); // 初始化串列通信
  dht.begin();        // 初始化 DHT11 傳感器
  pinMode(BUZZER_PIN, OUTPUT);
}

void loop() {
  // 檢查是否有來自 Jetson Nano 的指令
  if (Serial.available() > 0) {
    char command = Serial.read();  // 接收來自 Jetson Nano 的指令
    if (command == 'S') {
        noTone(BUZZER_PIN);
    } else {
        switch (command) {
          case 'M':  // Medication 音樂
            playMedicationTone();
            break;
          case 'B':  // Blood Pressure 音樂
            playBloodPressureTone();
            break;
          case 'E':  // Exercise 音樂
            playExerciseTone();
            break;
        }
      }
  }
  

  // 讀取溫濕度資料
  float temperature = dht.readTemperature(); // 讀取溫度
  float humidity = dht.readHumidity();       // 讀取濕度

  // 檢查是否讀取成功
  if (isnan(temperature) || isnan(humidity)) {
    Serial.println("{\"error\": \"Failed to read from DHT sensor!\"}");
  } else {
    // 傳遞結構化數據（JSON 格式）
    Serial.print("{\"temperature\": ");
    Serial.print(temperature);
    Serial.print(", \"humidity\": ");
    Serial.print(humidity);
    Serial.println("}");
  }

  delay(2000); // 每 2 秒讀取一次溫濕度
}

// 播放 "do do do re mi"
void playMedicationTone() {
  playNote(NOTE_DO, 300);
  playNote(NOTE_DO, 300);
  playNote(NOTE_DO, 300);
  playNote(NOTE_RE, 300);
  playNote(NOTE_MI, 300);
}

// 播放 "mi mi mi fa sol"
void playBloodPressureTone() {
  playNote(NOTE_MI, 300);
  playNote(NOTE_MI, 300);
  playNote(NOTE_MI, 300);
  playNote(NOTE_FA, 300);
  playNote(NOTE_SOL, 300);
}

// 播放 "fa fa sol la mi"
void playExerciseTone() {
  playNote(NOTE_FA, 300);
  playNote(NOTE_FA, 300);
  playNote(NOTE_SOL, 300);
  playNote(NOTE_LA, 300);
  playNote(NOTE_MI, 300);
}

// 播放單個音符的函數
void playNote(int frequency, int duration) {
  tone(BUZZER_PIN, frequency, duration);
  delay(duration + 100); // 確保音符之間有間隔
}
