import cv2 
from mtcnn.mtcnn import MTCNN
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as img
from os import listdir
import datetime
import os

detector = MTCNN()
forder = "./dataset/persons_data"   # forder chứa ảnh cần xác định face
wrong_detects = 0               # biến đếm ảnh mà mtcnn k phát hiện được face
total_image = 0               # chỉ là biến đặt tên ảnh cần lưu
wrong = []
print("start time")
print(datetime.datetime.now().time())  # xác định thời gian bắt đầu của model
for forder_name in listdir(forder):
    os.mkdir(forder + "/" + forder_name)
    print(forder_name)
    for filename in listdir(forder + "/"+ forder_name):
        cnt = np.random.rand(1)[0]
        total_image += 1
        path = forder + "/" + forder_name + "/" + filename
        image = cv2.imread(path)
        result = detector.detect_faces(image)
        if len(result) == 0:
          wrong_detects += 1
          wrong.append(filename)
          continue
        else:
          for person in result:
            bounding_box = person['box']
            keypoints = person['keypoints']
            im_crop = image[bounding_box[1] : bounding_box[1] + bounding_box[3], bounding_box[0]: bounding_box[0]+bounding_box[2] ]
            # print(im_crop.shape)
            if im_crop.shape[0] > 0 and im_crop.shape[1] > 0:
              cv2.imwrite(forder + "/" + forder_name + "/" + filename, im_crop)
    print("end")
print("end time")
print(datetime.datetime.now().time())
print("wrong", wrong)
print("wrong detects:",wrong_detects)
print("total images:", total_image)
print("accurance:",1-wrong_detects/total_image)