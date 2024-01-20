import cv2
import numpy as np


# Step1. RGB转换为HSV
img = cv2.imread('C:/Users/Alvin Ang/Desktop/zhinengwurenji/task3_3/6.jpg')
img = cv2.resize(img,(480,360))
#   显示原图 
cv2.imshow('result1', img)
cv2.waitKey(0)

hue_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Step2. 用颜色分割图像（red）
low_range = np.array([0, 0,0])
high_range = np.array([180, 255, 30])
th = cv2.inRange(hue_image, low_range, high_range)
cv2.imshow('result2', th)
cv2.waitKey(0)