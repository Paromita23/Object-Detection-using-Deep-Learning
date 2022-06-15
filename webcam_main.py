# See PyCharm help at https://www.jetbrains.com/help/pycharm/
import cv2
import numpy as np

thres = 0.5  # threshold to detect the objects
nms_thres = 0.2     #Non Maximum Suppression

#opening of webcam.
cap = cv2.VideoCapture(0)
cap.set(3, 720)     #width
cap.set(4, 640)     #height
cap.set(10, 150)    #brightness

classNames = []
classFile = 'dataset\\coco.names'

with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
#print(classNames)

#color mapping randomly.
#colors = np.random.uniform(0, 255, size=(len(classNames), 2))

configPath = 'C:\\Users\\PAROMITA SAHA\\PycharmProjects\\ObjectDetector\\dataset\\ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'C:\\Users\\PAROMITA SAHA\\PycharmProjects\\ObjectDetector\\dataset\\frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)


while True:
    success, img = cap.read()
    classIds, confs, bbox = net.detect(img, confThreshold=thres)
    #print(classIds, bbox)

    #bbox = list(bbox)
    #confs = list(np.array(confs).reshape(1, -1)[0])
    #confs = list(map(float, confs))
    #print(type(confs[0]))
    #print(confs)

    indices = cv2.dnn.NMSBoxes(bbox, confs, thres, nms_thres)
    print(indices)

    for i in range(len(bbox)):
        for i in indices:
            box = bbox[i]
            confidence = confs[i]
            x, y, w, h = box[0], box[1], box[2], box[3]

            conf = str(round(confidence * 100, 2)) + "%"

            cv2.rectangle(img, (x, y), (x+w, h+y), color=(230, 0, 0), thickness=3)
            cv2.putText(img, classNames[classIds[i]-1], (box[0] + 10, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 252), 2)
            cv2.putText(img, conf, (box[0] + 10, box[1] + 60), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)

            #cv2.rectangle(img, (x, y), (x + w, h + y), colors[classIds[i]-1], thickness=3)
            #cv2.putText(img, classNames[classIds[i] - 1], (box[0] + 10, box[1] + 30),
            #            cv2.FONT_HERSHEY_COMPLEX, 1, colors[classIds[i]-1], 2)
            #cv2.putText(img, conf, (box[0] + 10, box[1] + 60), cv2.FONT_HERSHEY_COMPLEX, 1, colors[classIds[i]-1], 2)

            print(classIds, classNames[classIds[i] - 1], conf)

    #output player pop up window
    cv2.imshow("Output_Video_Detection", img)
    cv2.waitKey(1)
