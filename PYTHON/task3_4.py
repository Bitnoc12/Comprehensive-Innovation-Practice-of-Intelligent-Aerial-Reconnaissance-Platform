import cv2
import numpy as np


# Step1. RGB转换为HSV
img = cv2.imread('C:/Users/Alvin Ang/Desktop/zhinengwurenji/task3_4/5.jpg')
img = cv2.resize(img,(480,360))
hue_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Step2. 用颜色分割图像（red）
low_range = np.array([0, 100, 100])
high_range = np.array([10, 255, 255])
th = cv2.inRange(hue_image, low_range, high_range)

# Step3. 膨胀图像
dilated = cv2.dilate(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4)), iterations=2)

# Step4. 霍夫圆
circles = cv2.HoughCircles(dilated, cv2.HOUGH_GRADIENT, 1, 100, param1=18, param2=6, minRadius=10, maxRadius=100)
circles = np.uint16(np.around(circles))
# Step5. 绘制圆形
if circles is not None:
    x, y, radius = circles [0][0]
    center = (x, y)
    cv2.circle(img, center, radius, (0, 255, 0), 2)
    cv2.circle(img, (x,y), 2, (0, 255, 0), 3)     # 画圆心
print("圆心坐标",center)
print('半径',radius)
cv2.imshow('result', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
