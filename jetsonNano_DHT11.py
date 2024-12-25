import serial
import json

# 打開 Arduino 的串口
arduino = serial.Serial('/dev/ttyACM0', 9600)

print("Listening for data from Arduino...")

try:
    while True:
        if arduino.in_waiting > 0:
            raw_data = arduino.readline().decode('utf-8').strip()
            try:
                # 解析 JSON 數據
                data = json.loads(raw_data)
                if "temperature" in data and "humidity" in data:
                    temperature = data["temperature"]
                    humidity = data["humidity"]
                    print(f"Temperature: {temperature} °C, Humidity: {humidity} %")
                else:
                    print("Invalid data:", data)
            except json.JSONDecodeError:
                print("Failed to decode JSON:", raw_data)
except KeyboardInterrupt:
    print("Program stopped.")
finally:
    arduino.close()
