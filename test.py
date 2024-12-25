import time
from datetime import datetime , timedelta
import threading
import Jetson.GPIO as GPIO
from PIL import Image, ImageDraw, ImageFont
import Adafruit_SSD1306
import serial

# 初始化串列通信
ser = serial.Serial('/dev/ttyACM0', 9600)  # 根據實際情況修改端口名稱


# 硬體配置
BUZZER_PIN = 18  # 蜂鳴器 GPIO
#BUTTON_PIN = 18  # 按鈕 GPIO
RST = None  # OLED 的 reset 設為 None
I2C_BUS = 1  # I2C 總線號


# 初始化 GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(BUZZER_PIN, GPIO.OUT)

#GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

# 初始化 OLED
disp = Adafruit_SSD1306.SSD1306_128_32(rst=RST, i2c_bus=I2C_BUS, gpio=1)
disp.begin()
disp.clear()
disp.display()

# 建立 OLED 繪製畫布
width = disp.width
height = disp.height
image = Image.new('1', (width, height))
draw = ImageDraw.Draw(image)
font = ImageFont.load_default()

# 定義事件時間
EVENTS = {
    "medication": {"time": ["12:57", "19:23", "19:25"], "music": "medication"},
    "blood pressure": {"time": ["12:55"], "music": "blood_pressure"},
    "exercise": {"time": ["12:59"], "music": "exercise"},
}


music_type = None
# 播放音樂函數
def play_music():
    global playing_event
    global music_type
    print(f"Sending {music_type} command to Arduino...")
    while True:
        print('here!!!')
        while playing_event:
          print(f"Sending {music_type} command to Arduino...")
          if music_type == "medication":
              for i in range(1):
                  print('M')
                  ser.write(b'M')  # 傳送 'M' 表示 medication
          elif music_type == "blood_pressure":
              for i in range(1):
                  ser.write(b'B')  # 傳送 'B' 表示 blood pressure
                  print('B')
          elif music_type == "exercise":
              for i in range(1):
                  ser.write(b'E')  # 傳送 'E' 表示 exercise
                  print('E')
          else:
              print("Unknown music type!")
          time.sleep(6)  # 給 Arduino 一些處理時間
        #ser.write(b'S')

# 顯示文字到 OLED
def display_on_oled(text_lines):
    draw.rectangle((0, 0, width, height), outline=0, fill=0)  # 清除畫面
    for i, line in enumerate(text_lines):
        draw.text((0, i * 8), line, font=font, fill=255)  # 每行占 8 像素高度
    disp.image(image)
    disp.display()

# 清空 OLED 顯示
def clear_oled():
    disp.clear()
    disp.display()

# 更新時間函數
def update_time_on_oled():
    global event_displaying
    while True:
        if not event_displaying:  # 當沒有提醒事件時，顯示時間
            current_time = time.strftime("%H:%M")
            draw.rectangle((0, 0, width, height), outline=0, fill=0)  # 清除畫面
            draw.text((0, 12), f"Time: {current_time}", font=font, fill=255)  # 顯示當前時間
            disp.image(image)
            disp.display()
        time.sleep(1)  # 每秒更新一次
time_format="%H:%M"

def notice_event():
    global playing_event, event_displaying, music_type
    playing_event = False
    event_displaying = False

    # 啟動時間顯示執行緒
    #time_thread = threading.Thread(target=update_time_on_oled, daemon=True)
    #time_thread.start()

    # 播放音樂執行緒
    music_thread = threading.Thread(target=play_music,args=())
    music_thread.start()

    try:
        while True:
            # 取得目前時間
            current_time = datetime.strptime(time.strftime("%H:%M"), time_format)
            # ser.write(b'S')  # 傳送 'M' 表示 medication
            # 檢查是否有事件
            for event, details in EVENTS.items():
                for time_str in details["time"]:
                    # 將時間字串轉換為 datetime
                    event_time = datetime.strptime(time_str, time_format)

                    # 如果目前時間等於事件時間，觸發事件
                    if current_time == event_time:
                        playing_event = True
                        event_displaying = True
                        text_lines = [f"Event:", f"{event}"]
                        display_on_oled(text_lines)

                        music_type = details["music"]
                        print(music_type)
                        print(f"Event: {event} started.")
                        print(playing_event)

                    # 如果目前時間超過事件時間 1 分鐘，重設狀態
                    elif current_time > event_time: #+ timedelta(minutes=1):
                        if playing_event:  # 確保這裡是處理中的事件
                            print(f"Event {event} ended (timeout).")
                            playing_event = False  # 停止播放音樂
                            event_displaying = False  # 停止事件顯示
                            clear_oled()  # 清除 OLED 顯示
                            # ser.write(b'S')
                            # print('S')
            #time.sleep(1)

    except KeyboardInterrupt:
        print("Program terminated.")
    finally:
        GPIO.cleanup()
# 主程序
def main():

    notice_event()

if __name__ == "__main__":
    main()
