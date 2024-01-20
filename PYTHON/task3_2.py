import time
import numpy as np
import cv2

#读入图像
img = cv2.resize(cv2.imread('C:/Users/Alvin Ang/Desktop/zhinengwurenji/task3_2/6.jpg'),(480,360))
imggary=cv2.resize(cv2.imread("C:/Users/Alvin Ang/Desktop/zhinengwurenji/task3_2/6.jpg",cv2.IMREAD_GRAYSCALE),(480,360))#灰度图


#显示原始图像
print("原始")
cv2.imshow('initial pic',img)
cv2.waitKey(0)

### K-means均值
#中值滤波
medianBlur_result1 = cv2.medianBlur(imggary, 13)
print("中值滤波")  
cv2.imshow('medianBlur_result1', medianBlur_result1)
cv2.waitKey(0)

#均值自适应阈值分割
adaptive_threshold_img = cv2.adaptiveThreshold(medianBlur_result1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 85, 20)
print("均值自适应阈值分割")
cv2.imshow('adaptive_threshold_img', adaptive_threshold_img)
cv2.waitKey(0) 

##霍夫变换
circle = cv2.HoughCircles(adaptive_threshold_img,cv2.HOUGH_GRADIENT,1,200,param1=50,param2=25,minRadius=50,maxRadius=180) 
try:
    img_new=adaptive_threshold_img
    circles = circle.reshape(-1, 3)
    circles = sorted(np.uint16(np.around(circles)), key=lambda x: x[2], reverse=True)
    r = circles[0][2]
    x = circles[0][0]
    y = circles[0][1]
    

    #掩码
    #行
    row1=y-r-8
    row2=y+r+8
    #列
    col1=x-r-8
    col2=x+r+8
        
    mask = np.zeros(img_new.shape, np.uint8)
    mask.fill(255)

    mask[int(row1):int(row2),int(col1):int(col2)]=0

    #提取框选部分
    img_new=cv2.bitwise_or(img_new,mask)
        
except:
    print('change param!')

#中值滤波
medianBlur_result2 = cv2.medianBlur(img_new, 9)
print("中值滤波")
cv2.imshow('medianBlur_result2', medianBlur_result2)
cv2.waitKey(0)

#边缘检测
canny = cv2.Canny(medianBlur_result2, 100, 300,L2gradient=True )
print("边缘检测")
cv2.imshow('canny', canny)
cv2.waitKey(0)

#膨胀
dilate_kernel = np.ones((5,5), np.uint8)
dilate_image = cv2.dilate(canny, dilate_kernel, iterations=2)
print("膨胀")
cv2.imshow('dilate_image', dilate_image)
cv2.waitKey(0)

##霍夫变换
circle = cv2.HoughCircles(dilate_image,cv2.HOUGH_GRADIENT,1,150,param1=300,param2=25,minRadius=30,maxRadius=180) 
try:
    circles = circle.reshape(-1, 3)
    circles = sorted(np.uint16(np.around(circles)), key=lambda x: x[2], reverse=True)
    r = circles[0][2]
    x = circles[0][0]
    y = circles[0][1]
    print("圆心坐标为：", (x, y))
    print("圆的半径是：", r)

    cv2.circle(img, (x,y), r, (0, 0, 255), 5)   # 画圆
    cv2.circle(img, (x,y), 2, (0, 255, 0), 10)     # 画圆心

    print("霍夫变换")
    cv2.imshow('hough', img)
    cv2.waitKey(0)
except:
    print('change param!')
