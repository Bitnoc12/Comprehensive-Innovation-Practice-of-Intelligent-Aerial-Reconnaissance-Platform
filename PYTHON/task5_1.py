from djitellopy import tello
import cv2
import numpy as np

me = tello.Tello()
me.connect()
print(me.get_battery()) #电池的电量
me.streamon()

while True:
    img = me.get_frame_read().frame
    img = cv2.resize(img,(480,360))
    
    imggary = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  #灰度

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
        r = circles[0][2]
        x = circles[0][0]
        y = circles[0][1]

        #掩码
        #行
        row1=y-r-10
        row2=y+r+10
        #列
        col1=x-r-10
        col2=x+r+10
        
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
    canny = cv2.Canny(medianBlur_result2, 100, 300,L2gradient=True )

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

        cv2.circle(img, (x,y), 2, (0, 255, 0), 10)     # 画圆心
        cv2.circle(img, (x,y), r, (0, 0, 255), 5)   # 画圆
    
    except:
        print('change param')

    cv2.imshow("image",img)
    cv2.waitKey(25)
