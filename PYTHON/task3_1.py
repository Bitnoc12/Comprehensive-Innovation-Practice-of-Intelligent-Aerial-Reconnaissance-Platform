import cv2
import numpy as np

img = 'C:/Users/Alvin Ang/Desktop/zhinengwurenji/task31/3.jpg'

def circleDetec(img):
    
    img = cv2.imread(img)
    image=img
    # 均值迁移滤波
    filter = cv2.pyrMeanShiftFiltering(image, 10, 100)
    # 转换成灰度图
    img_gray = cv2.cvtColor(filter, cv2.COLOR_BGR2GRAY)
    #print(img_gray.shape)
    
    circle = cv2.HoughCircles(img_gray,cv2.HOUGH_GRADIENT,0.5,100,param1=50,param2=20,minRadius=30,maxRadius=100) 
    try:
        circles = circle.reshape(-1, 3)
    except:
        print('change param!')
        return
    circles = np.uint16(np.around(circles))

    for i in circles:
        cv2.circle(img, (i[0], i[1]), i[2], (255, 0, 0), 5)   # 画圆
        cv2.circle(img, (i[0], i[1]), 2, (0, 255, 0), 10)     # 画圆心

    #圆数据
    print("圆的个数是：")
    print(len(circles))
    for i in circles:
        r = i[2]
        x = i[0]
        y = i[1]
        print("圆心坐标为：", (x, y))
        print("圆的半径是：", r)

    
    cv2.imshow('', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

circleDetec(img)

