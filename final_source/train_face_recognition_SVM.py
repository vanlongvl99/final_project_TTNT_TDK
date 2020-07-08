import cv2 
import os 
import numpy as np 
from mtcnn.mtcnn import MTCNN
from sklearn.svm import SVC
from tensorflow.keras.models import load_model
import joblib
from numpy import expand_dims
from sklearn.preprocessing import Normalizer
import datetime
from numpy import load
from os import listdir 

path = './dataset/emotion_data'
    label_to_index = {}
    cnt = 0
    for forder_name in listdir(path):
        label_to_index[forder_name] = cnt
        cnt += 1
    # print(label_to_index)
    X_train = []
    y_train = []
    print(datetime.datetime.now())
    for forder_name in listdir(path):
        count = 0
        for file_name in listdir(path + '/' + forder_name):
            path_of_image = path + '/' + forder_name + '/' + file_name
            image = cv2.imread(path_of_image)
            # print(count)
            count += 1
            if count == 200:
                break
            image = cv2.resize(image,(160,160))
            image = image/255
            X_train.append(image)
            y_train.append(label_to_index[forder_name])
        print( forder_name,count)
    print(datetime.datetime.now())
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    # convert each face in the train set to an embedding
    newX_train = list()import cv2 
import os 
import numpy as np 
from mtcnn.mtcnn import MTCNN
from sklearn.svm import SVC
from tensorflow.keras.models import load_model
import joblib
from numpy import expand_dims
from sklearn.preprocessing import Normalizer
import datetime
from numpy import load
from os import listdir 

path = './dataset/face_data'
    label_to_index = {}
    cnt = 0
    for forder_name in listdir(path):
        label_to_index[forder_name] = cnt
        cnt += 1
    # print(label_to_index)
    X_train = []
    y_train = []
    print(datetime.datetime.now())
    for forder_name in listdir(path):
        count = 0
        for file_name in listdir(path + '/' + forder_name):
            path_of_image = path + '/' + forder_name + '/' + file_name
            image = cv2.imread(path_of_image)
            # print(count)
            count += 1
            if count == 200:
                break
            image = cv2.resize(image,(160,160))
            image = image/255
            X_train.append(image)
            y_train.append(label_to_index[forder_name])
        print( forder_name,count)
    print(datetime.datetime.now())
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    # convert each face in the train set to an embedding
    newX_train = list()
    for face_pixels in X_train:
    	embedding = get_embedding(model_facenet, face_pixels)
    	newX_train.append(embedding)
    newX_train = np.array(newX_train)
    print(newX_train.shape)
    print(datetime.datetime.now())
    # normalize input vectors
    in_encoder = Normalizer(norm='l2')
    newX_train = in_encoder.transform(newX_train)
    # fit model
    model_svm = SVC(kernel='linear', probability=True)
    model_svm.fit(newX_train, y_train)
    # predict
    yhat_train = model_svm.predict(newX_train)
    print(datetime.datetime.now(), "finished")
    #save model svm
    filename = './model_file/face_recognition_model_svm_1.sav'
    joblib.dump(model_svm, filename)
  
    for face_pixels in X_train:
    	embedding = get_embedding(model_facenet, face_pixels)
    	newX_train.append(embedding)
    newX_train = np.array(newX_train)
    print(newX_train.shape)
    print(datetime.datetime.now())
    # normalize input vectors
    in_encoder = Normalizer(norm='l2')
    newX_train = in_encoder.transform(newX_train)
    # fit model
    model_svm = SVC(kernel='linear', probability=True)
    model_svm.fit(newX_train, y_train)
    # predict
    yhat_train = model_svm.predict(newX_train)
    print(datetime.datetime.now(), "finished")
    #save model svm
    filename = './model_file/emotion_recognition_model_svm_1.sav'
    joblib.dump(model_svm, filename)
  