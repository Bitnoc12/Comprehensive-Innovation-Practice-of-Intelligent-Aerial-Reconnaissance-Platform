from djitellopy import tello
import cv2
import numpy as np

me = tello.Tello()
me.connect()
print(me.get_battery()) #电池的电量
me.streamon()

def circle(image):
    img = image

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
            x, y, r = circles [0][0]
            center = (x, y)
            cv2.circle(img, center, r, (0, 255, 0), 2)
            cv2.circle(img, (x,y), 2, (0, 255, 0), 3)     # 画圆心
        print("圆心坐标",center)
        print('半径',r)
    except:
        print('try again')
    return img


while True:
    img = me.get_frame_read().frame
    img = cv2.resize(img, (480, 360))

    img = circle(img)

    cv2.imshow("image", img)
    cv2.waitKey(25)



