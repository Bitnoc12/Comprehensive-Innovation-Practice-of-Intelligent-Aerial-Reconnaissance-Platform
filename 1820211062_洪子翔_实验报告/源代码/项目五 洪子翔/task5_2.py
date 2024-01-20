from djitellopy import tello
import cv2
import numpy as np
import math
from time import sleep

me = tello.Tello()
me.connect()
print(me.get_battery()) #电池的电量
me.streamon()
me.takeoff()
me.get_height()
me.move_up(68)
me.flip_right()



x,y,r=0,0,0
center_x,center_y=245,122

def circle(image):
    global x,y,r
    imggary = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)  #灰度

    ### K-means均值
    #中值滤波
    medianBlur_result1 = cv2.medianBlur(imggary, 13)

    #均值自适应阈值分割
    adaptive_threshold_img = cv2.adaptiveThreshold(medianBlur_result1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 85, 20)

    ##霍夫变换
    circle = cv2.HoughCircles(adaptive_threshold_img,cv2.HOUGH_GRADIENT,1,200,param1=50,param2=25,minRadius=50,maxRadius=180) 
    try:
        img_new=adaptive_threshold_img
        circles = circle.reshape(-1, 3)
        circles = sorted(np.uint16(np.around(circles)), key=lambda x: x[2], reverse=True)
        r = int(circles[0][2])
        x = int(circles[0][0])
        y = int(circles[0][1])

        #掩码
        #行
        row1=0 if y-r-10<0 else y-r-10
        row2=360 if y+r+10>360 else y+r+10
        #列
        col1=0 if x-r-10<0 else x-r-10
        col2=480 if x+r+10>480 else x+r+10
        
        mask = np.zeros(img_new.shape, np.uint8)
        mask.fill(255)

        mask[int(row1):int(row2),int(col1):int(col2)]=0

        #提取框选部分
        img_new=cv2.bitwise_or(img_new,mask)
        
    except:
        print('')

    #中值滤波
    medianBlur_result2 = cv2.medianBlur(img_new, 9)

    #边缘检测
    canny = cv2.Canny(medianBlur_result2, 100, 300,L2gradient=True)

    #膨胀
    dilate_kernel = np.ones((5,5), np.uint8)
    dilate_image = cv2.dilate(canny, dilate_kernel, iterations=1)

    ##霍夫变换
    circle = cv2.HoughCircles(dilate_image,cv2.HOUGH_GRADIENT,1,150,param1=50,param2=25,minRadius=50,maxRadius=180) 
    try:
        circles = circle.reshape(-1, 3)
        #print(circles)
        circles = sorted(np.uint16(np.around(circles)), key=lambda x: x[2], reverse=True)
        r = circles[0][2]
        x = circles[0][0]
        y = circles[0][1]
        print("圆心坐标为：", (x, y))
        print("圆的半径是：", r)

        cv2.circle(image, (x,y), 2, (0, 255, 0), 10)     # 画圆心
        cv2.circle(image, (x,y), r, (0, 0, 255), 5)   # 画圆
    
    except:
        print('LAND!!!')

    return image,x,y

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
        me.move_forward(300)
        me.rotate_clockwise(180)
        me.move_up(40)
        me.move_forward(300)
        me.flip_right()
        me.rotate_clockwise(90)
        me.land()
    

    cv2.imshow("image", img)
    if cv2.waitKey(25) & 0xff == 27:    #紧急降落——按下Esc降落
        me.land()
        break
