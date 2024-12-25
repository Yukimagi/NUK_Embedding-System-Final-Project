import tensorrt as trt
import pycuda.driver as cuda
import numpy as np
import pycuda.autoinit
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import tensorrt as trt
import argparse
import yaml
import time
import pafy
import threading

import Adafruit_SSD1306   # This is the driver chip for the Adafruit PiOLED

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

import subprocess
import Adafruit_DHT
import requests

from test import notice_event
import serial
import json


fall_count = 0
recover_count = 0
FALL_THRESHOLD = 30
RECOVER_THRESHOLD = 30
HUMIDITY_THRESHOLD = 70

# Line Notify 權杖
LINE_NOTIFY_TOKEN = 'QN4jKNcZpNUNyAAcfj8M62lIiALBto8LdRwcXrEOBuc'
notified_fall = False
notified_DHT = False

# DHT pin
DHT_PIN = 13
DHT_TYPE = Adafruit_DHT.DHT11

# DHT
humidity = 0
temperature = 0


# 發送 Line Notify 訊息
def send_line_notify(message):
    url = 'https://notify-api.line.me/api/notify'
    headers = {'Authorization': f'Bearer {LINE_NOTIFY_TOKEN}'}
    data = {'message': message}
    response = requests.post(url, headers=headers, data=data)
    return response.status_code

# 讀取 DHT11 溫濕度
'''
def read_dht11_data():
    humidity, temperature = Adafruit_DHT.read_retry(DHT_TYPE, DHT_PIN)
    return humidity, temperature
'''


def DHT11():
    global temperature, humidity, notified_DHT
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
                        
                        if (temperature > 30 or humidity > 70) and not notified_DHT:
                            send_line_notify(f'溫濕度異常通知⚠️ 溫度：{temperature}, 濕度：{humidity}')
                            notified_DHT = True

                        else:
                            notified_DHT = False
                    else:
                        print("Invalid data:", data)
                except json.JSONDecodeError:
                    print("Failed to decode JSON:", raw_data)
    except KeyboardInterrupt:
        print("Program stopped.")
    finally:
        arduino.close()


def non_maximum_suppression_fast(boxes, overlapThresh=0.3):
    
    # If there is no bounding box, then return an empty list
    if len(boxes) == 0:
        return []
        
    # Initialize the list of picked indexes
    pick = []
    
    # Coordinates of bounding boxes
    x1 = boxes[:,0].astype("float")
    y1 = boxes[:,1].astype("float")
    x2 = boxes[:,2].astype("float")
    y2 = boxes[:,3].astype("float")
    
    # Calculate the area of bounding boxes
    bound_area = (x2-x1+1) * (y2-y1+1)
    # Sort the bounding boxes by the bottom-right y-coordinate of the bounding box
    sort_index = np.argsort(y2)
    
    
    # Looping until nothing left in sort_index
    while sort_index.shape[0] > 0:
        # Get the last index of sort_index
        # i.e. the index of bounding box having the biggest y2
        last = sort_index.shape[0]-1
        i = sort_index[last]
        
        # Add the index to the pick list
        pick.append(i)
        
        # Compared to every bounding box in one sitting
        xx1 = np.maximum(x1[i], x1[sort_index[:last]])
        yy1 = np.maximum(y1[i], y1[sort_index[:last]])
        xx2 = np.minimum(x2[i], x2[sort_index[:last]])
        yy2 = np.minimum(y2[i], y2[sort_index[:last]])        

        # Calculate the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # Compute the ratio of overlapping
        overlap = (w*h) / bound_area[sort_index[:last]]
        
        # Delete the bounding box with the ratio bigger than overlapThresh
        sort_index = np.delete(sort_index, 
                               np.concatenate(([last], np.where(overlap > overlapThresh)[0])))
        
    # return only the bounding boxes in pick list        
    # return boxes[pick]
    return pick

def load_engine(trt_runtime, plan_path):

    engine = trt_runtime.deserialize_cuda_engine(Path(plan_path).read_bytes())
    return engine

def allocate_buffers(engine, batch_size):

    inputs = []
    outputs = []
    bindings = []
    # data_type = engine.get_binding_dtype(0)

    for binding in engine:
        # print(engine.get_binding_dtype(binding))
        size = trt.volume(engine.get_binding_shape(binding)) * batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        
        host_mem = cuda.pagelocked_empty(size, dtype=dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))

        dic = {
                "host_mem" : host_mem,
                "device_mem" : device_mem,
                "shape" : engine.get_binding_shape(binding),
                "dtype" : dtype
            }
        if engine.binding_is_input(binding):
            inputs.append(dic)
        else:
            outputs.append(dic)

    stream = cuda.Stream()
    return inputs , outputs , bindings , stream

def load_images_to_buffer(pics, pagelocked_buffer):
   
   preprocessed = np.asarray(pics).ravel()
   np.copyto(pagelocked_buffer, preprocessed)

def do_inference(context, pics_1, inputs , outputs , bindings , stream, model_output_shape):

    start = time.perf_counter()
    load_images_to_buffer(pics_1, inputs[0]["host_mem"])

    [cuda.memcpy_htod_async(intput_dic['device_mem'], intput_dic['host_mem'], stream) for intput_dic in inputs]

    # Run inference.

    # context.profiler = trt.Profiler()
    context.execute(batch_size=1, bindings=bindings)

    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(output_dic["host_mem"], output_dic["device_mem"], stream) for output_dic in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return the host output.
    out = outputs[0]["host_mem"].reshape((outputs[0]['shape']))
    # out = h_output

    return out , time.perf_counter() - start


def draw_detect(img , x1 , y1 , x2 , y2 , conf , class_id , label , color_palette):
    global notified_fall, humidity, temperature
    # label = f'{CLASSES[class_id]} ({confidence:.2f})'
    color = color_palette[class_id]
    
    # print(x1 , y1 , x2 , y2 , conf , class_id)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    
    height = abs(y2 - y1)
    wideth = abs(x2 - x1)

    if wideth >= height - 2:
        cv2.putText(img, f"Falling Detect", (x1 - 10, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        if not notified_fall:
                message = "滑倒⚠️：你是不是滑倒了？" if humidity >= HUMIDITY_THRESHOLD else "警告⚠️：偵測到跌倒！"
                # message = "滑倒⚠️：你是不是滑倒了？"
                send_line_notify(message)
                notified_fall = True

    else:
        cv2.putText(img, f"{label[class_id]} Non Falling", (x1 - 10, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def show_detect(img , preds , iou_threshold , conf_threshold, class_label , color_palette):
    boxes = []
    scores = []
    class_ids = []
    
    max_conf = np.max(preds[0,4:,:] , axis=0)
    idx_list = np.where(max_conf > conf_threshold)[0]
    
    for pred_idx in idx_list:

        pred = preds[0,:,pred_idx]
        conf = pred[4:]
        
        
        box = [pred[0] - 0.5*pred[2], pred[1] - 0.5*pred[3] , pred[0] + 0.5*pred[2] , pred[1] + 0.5*pred[3]]
        boxes.append(box)

        label = np.argmax(conf)
        
        scores.append(max_conf[pred_idx])
        class_ids.append(label)

    boxes = np.array(boxes)
    result_boxes = non_maximum_suppression_fast(boxes, overlapThresh=iou_threshold)
    

    for i in range(len(result_boxes)):
        index = result_boxes[i]
        box = boxes[index]
        
        draw_detect(img, round(box[0]), round(box[1]),round(box[2]), round(box[3]),
            scores[index] , class_ids[index] , class_label , color_palette)
    
    return [{"boxes" : boxes[i] , "class_id" : class_ids[i] , "conf" : scores[i]} for i in result_boxes]
        

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs=1, type=str, help='model path')
    parser.add_argument('--source', nargs=1 , type=str  ,help='inference target')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--data', nargs=1 , type=str, help=' dataset.yaml path')
    parser.add_argument('--show', action="store_true", help=' show detect result')

    opt = parser.parse_args()
    return opt

def main(opt):
    print(opt)
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    trt_runtime = trt.Runtime(TRT_LOGGER)
    engine_path = opt['weights'][0]
    engine = load_engine(trt_runtime, engine_path)
    source =  opt['source'][0]
    iou_threshold =  opt['iou_thres']
    conf_threshold = opt['conf_thres']
    yaml_path = opt['data'][0]
    show = opt['show']

    with open(yaml_path, 'r') as stream:
        data = yaml.load(stream)
    
    label = data['names']
    color_palette = np.random.uniform(0, 255, size=(len(label), 3))
    print(label)

    video_inferences(source , engine , iou_threshold , conf_threshold , label , color_palette , show)



def video_inferences(video_path , engine , iou_threshold , conf_threshold , label , color_palette , show):
    inputs , outputs , bindings , stream = allocate_buffers(engine, 1)
    context = engine.create_execution_context()

    WIDTH = inputs[0]["shape"][2]
    HEIGHT = inputs[0]["shape"][3]

    model_output_shape = outputs[0]['shape']

    video_info = "video"

    if "youtube.com"  in video_path: 
        video_info = pafy.new(video_path)  
        video_path = video_info.getbest(preftype='mp4').url
    elif len(video_path.split('.')) == 1: 
        video_info = "webcam"
        video_path = int(video_path)
    
    print(f"Inference with : {video_info}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("VideoCapture Error")
        return
    class_counts = {}  # Dictionary to store class counts
    # initial OLED
    import Adafruit_SSD1306
    disp = Adafruit_SSD1306.SSD1306_128_32(rst=None, i2c_bus=1, gpio=1)
    disp.begin()
    disp.clear()
    disp.display()
    # Create blank image for drawing.
    # Make sure to create image with mode '1' for 1-bit color.
    width = disp.width
    height = disp.height
    image = Image.new('1', (width, height))

    # Get drawing object to draw on image.
    draw = ImageDraw.Draw(image)

    # Draw a black filled box to clear the image.
    draw.rectangle((0, 0, width, height), outline=0, fill=0)

    # Draw some shapes.
    # First define some constants to allow easy resizing of shapes.
    padding = -2
    top = padding
    bottom = height-padding
    # Move left to right keeping track of the current x position for drawing shapes.
    x = 0

    # Load default font.
    #font = ImageFont.load_default()
    font = ImageFont.load_default()
    line_height= 16 
    while(True):
        detect_results = {}
        ret, frame = cap.read()
        if not ret:
            break

        
        start_time = time.perf_counter()
        frame = cv2.resize(frame , (WIDTH , HEIGHT))            
        im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        im = np.array(im, dtype=np.float32, order='C')
        im = im.transpose((2, 0, 1))
        im /=  255
        out , infer_time = do_inference(context, im, inputs , outputs , bindings, stream, model_output_shape)
        total_time = time.perf_counter() - start_time
        
        detect_results = show_detect(frame , out , iou_threshold , conf_threshold , label , color_palette)
        # TODO show the FPS & class name & number of each class in frame on OLED
        ###########################
        # Calculate FPS
        # calculate classes

        ###########################
        ###########################

        

        if show:
            cv2.imshow("img" , frame)

        if cv2.waitKey(1) == ord('q'):
            break
            
    cv2.destroyAllWindows()


if __name__ == "__main__" :
    opt = parse_opt()
    notice_thread = threading.Thread(target=notice_event,args=())
    DHT11_thread = threading.Thread(target=DHT11, args=())
    notice_thread.start()
    DHT11_thread.start()
    main(vars(opt))

