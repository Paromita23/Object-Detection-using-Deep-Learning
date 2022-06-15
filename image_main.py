# See PyCharm help at https://www.jetbrains.com/help/pycharm/
import cv2
import numpy as np

#input taken as image
#img = cv2.imread('images/elsa1' + '.jpeg')
img = cv2.imread('images/pic (5)' + '.png')
#img1 = cv2.imread('images/refrigerator.png')

resize_img = cv2.resize(img, (512, 512))

classNames = []
classFile = 'dataset\\coco.names'


with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
#print(classNames)

#color mapping randomly.
colors = np.random.uniform(0, 255, size=(len(classNames), 2))

configPath = 'C:\\Users\\PAROMITA SAHA\\PycharmProjects\\ObjectDetector\\dataset\\ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'C:\\Users\\PAROMITA SAHA\\PycharmProjects\\ObjectDetector\\dataset\\frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

classIds, confs, bbox = net.detect(resize_img, confThreshold=0.5)
#classIds, confs, bbox = net.detect(img1, confThreshold=0.5)


for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):

    conf = str(round(confidence * 100, 2)) + "%"

    cv2.rectangle(resize_img, box, colors[classId - 1], thickness=4)
    cv2.putText(resize_img, classNames[classId-1], (box[0] + 10, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX, 1, colors[classId-1], 2)
    cv2.putText(resize_img, conf, (box[0] + 10, box[1] + 70), cv2.FONT_HERSHEY_COMPLEX, 1, colors[classId-1], 2)

    #cv2.rectangle(resize_img, box, color=(0, 215, 0), thickness=4)
    #cv2.putText(resize_img, classNames[classId-1], (box[0] + 10, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX, 1, (230, 0, 0), 2)
    #cv2.putText(resize_img, conf, (box[0] + 10, box[1] + 70), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 215, 255), 2)

    print(classIds, classNames[classId-1], conf)

#for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
    #cv2.rectangle(img1, box, color=(230, 0, 0), thickness=3)
    #cv2.putText(img1, classNames[classId-1].upper(), (box[0] + 10, box[1] + 30),
    #            cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 250), 2)
    #cv2.putText(img1, str(round(confidence * 100, 2)) + "%", (box[0] + 10, box[1] + 60),
    #            cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 230, 0), 2)

cv2.imshow("Output_image_detection", resize_img)
#cv2.imshow("Output_image_detection", img1)
cv2.waitKey(0)
