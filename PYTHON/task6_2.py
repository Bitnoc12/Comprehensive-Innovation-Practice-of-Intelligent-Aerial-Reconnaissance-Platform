#导入
import numpy as np
import time
import cv2

# YOLO 对象检测
labelsPath = 'C:/Users/Alvin Ang/Desktop/zhinengwurenji/task6/coco.names'
config_path = 'C:/Users/Alvin Ang/Desktop/zhinengwurenji/task6/yolov4-tiny.cfg'
weights_path = 'C:/Users/Alvin Ang/Desktop/zhinengwurenji/task6/yolov4-tiny.weights'
net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

image_path='C:/Users/Alvin Ang/Desktop/zhinengwurenji/task6/6.jpg'
confidence_t=0.2    #置信区间
threshold=0.9   #阈值
# 加载训练 YOLO 模型
LABELS = open(labelsPath).read().strip().split("\n")
target_lables = ['person','bottle','cup']
# 初始化一个颜色列表来表示每个类标签
np.random.seed(19)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),dtype="uint8")


# 加载我们的输入图像并获取其空间维度
image = cv2.imread(image_path)
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
# 展示图像
cv2.imshow("Image", image)
cv2.waitKey(0)


