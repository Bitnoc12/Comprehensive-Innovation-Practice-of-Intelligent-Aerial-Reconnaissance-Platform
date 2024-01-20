from djitellopy import tello
import cv2
import math
import numpy as np
from time import sleep

me = tello.Tello()
me.connect()
print(me.get_battery()) #电池的电量
me.streamon()
me.takeoff()
me.move_up(60)


x, y, r = 0, 0, 0
center_x,center_y=240,156

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

while True:
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
        me.rotate_clockwise(90)
        me.move_forward(50)
        me.rotate_clockwise(90)
        me.move_forward(250)
        me.flip_forward()
        me.rotate_clockwise(360)
        me.streamoff()
        me.land()

    cv2.imshow("image", img)
    if cv2.waitKey(25) & 0xff == 27:    #紧急降落——按下Esc降落
        me.land()
        break



