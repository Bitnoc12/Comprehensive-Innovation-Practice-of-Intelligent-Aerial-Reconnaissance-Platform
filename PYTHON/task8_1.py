from djitellopy import tello
import cv2
import math
import time
import numpy as np
from time import sleep

me = tello.Tello()
me.connect()
print(me.get_battery()) #电池的电量
me.streamon()
me.takeoff()
me.move_up(60)


x, y, r = 0, 0, 0
center_x,center_y=242,156

#目标识别
throw_flag=0  #0表示未穿越，1表示已穿越
objectname=[]

# YOLO 对象检测
labelsPath = 'C:/Users/Alvin Ang/Desktop/zhinengwurenji/task6/coco.names'
config_path = 'C:/Users/Alvin Ang/Desktop/zhinengwurenji/task6/yolov4-tiny.cfg'
weights_path = 'C:/Users/Alvin Ang/Desktop/zhinengwurenji/task6/yolov4-tiny.weights'
net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
confidence_t=0.2    #置信区间
threshold=0.35   #阈值

# 标签
LABELS = open(labelsPath).read().strip().split("\n")
target_lables = ['person','bottle','cup']
# 初始化一个颜色列表来表示每个类标签
np.random.seed(19)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),dtype="uint8")

def circle(image):
    global x, y, r
    img = me.get_frame_read().frame
    img = cv2.resize(img,(480,360))

    # Step1. RGB转换为HSV
    hue_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Step2. 用颜色分割图像（red）
    low_range = np.array([0, 100, 100])
    high_range = np.array([10, 255, 255])
    th = cv2.inRange(hue_image, low_range, high_range)

    # Step3. 膨胀图像
    dilated = cv2.dilate(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2)

    try:
        # Step4. 霍夫圆
        circles = cv2.HoughCircles(dilated, cv2.HOUGH_GRADIENT, 1, 100, param1=15, param2=7, minRadius=10, maxRadius=100)
        circles = np.uint16(np.around(circles))
        # Step5. 绘制圆形
        if circles is not None:
            x, y, radius = circles [0][0]
            center = (x, y)
            cv2.circle(img, center, radius, (0, 255, 0), 2)
            cv2.circle(img, (x,y), 2, (0, 255, 0), 3)     # 画圆心
        print("圆心坐标",center)
        print('半径',radius)
    except:
        print('try again')
    return img, x, y

def fly_to_center(x,y,d):
    print("move")
    me.send_rc_control(int(8 * (int(x) - center_x) / d), 0, int(8 * (center_y - int(y)) / d), 0)
    sleep(1)

def yolo(image):
    # 加载我们的输入图像并获取其空间维度
    image = cv2.resize(image,(705,500))
    (H, W) = image.shape[:2]
    # 从输入图像构建一个blob，然后执行一个前向传播
    # 通过 YOLO 对象检测器，输出边界框和相关概率
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    # 获取网络输出层信息（所有输出层的名字），设定并前向传播
    outInfo = net.getUnconnectedOutLayersNames()
    # 得到各个输出层的、各个检测框等信息，是二维结构。
    layerOutputs = net.forward(outInfo)

    # 分别初始化检测到的边界框、置信度和类 ID 的列表
    boxes = []
    confidences = []
    classIDs = []
    # 循环输出
    for output in layerOutputs:
        # 遍历每个检测结果
        for detection in output:
            # 提取物体检测的类ID和置信度（即概率）
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            # 过滤精度低的结果
            if confidence > confidence_t:
                # 延展边界框坐标，计算 YOLO 边界框的中心 (x, y) 坐标，然后是框的宽度和高度
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                # 使用中心 (x, y) 坐标导出边界框的上角和左角
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                # 更新边界框坐标、置信度和类 ID 列表
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
    # 使用非极大值抑制来抑制弱的、重叠的边界框
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence_t,threshold)
    # 确保至少存在一个检测
    if len(idxs) > 0:
        # 遍历我们保存的索引
        for i in idxs.flatten():
            if LABELS[classIDs[i]] in target_lables:
                # 提取边界框坐标
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                # 在图像上绘制一个边界框矩形和标签
                color = [int(c) for c in COLORS[classIDs[i]]]
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)
                objectname.append(LABELS[classIDs[i]])

    return image

missionNum=0
while True:
    if missionNum==0:
      img = me.get_frame_read().frame
      img = cv2.resize(img, (480, 360))

      img, x, y = circle(img)

      d = int(math.sqrt((center_x - x) * (center_x - x) + (center_y - y) * (center_y - y)))
      print("dx=", x - center_x)
      print("dy=", y - center_y)

      if abs(x-center_x) > 15 or abs(y-center_y) > 15:
          fly_to_center(x, y, d)
      else:
          me.move_up(40)
          me.move_forward(250)
          me.move_down(80)
          missionNum+=1
      cv2.imshow("image", img)

      if cv2.waitKey(25) & 0xff == 27:    #紧急降落——按下Esc降落
          me.land()
          break
    elif missionNum==1:
      img = me.get_frame_read().frame
      img = cv2.resize(img, (480, 360))

      img=yolo(img)
      print(objectname)
      if len(objectname)==10:
          name=max(objectname,key=objectname.count)
          objectname=[]
          if name=='person':  #执行翻转动作需保证起飞电量大于60
              me.flip_right()
              time.sleep(1)
              me.land()
          elif name=='bottle':
              me.flip_back()
              time.sleep(1)
              me.land()
          elif name=='cup':
              me.flip_forward()
              time.sleep(1)
              me.land()
          else:
              me.flip_left()
              time.sleep(1)
              me.land()
            
      cv2.imshow("image", img)

      if cv2.waitKey(20) & 0xff == 27:    #紧急降落——按下Esc降落
          me.land()
          break


