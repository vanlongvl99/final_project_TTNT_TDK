from sklearn.model_selection import train_test_split
from keras.layers import Dense, Dropout, MaxPooling2D, Conv2D, Flatten
from keras import Sequential
import keras
import datetime
from os import listdir
import numpy as np
import cv2




label_to_index = {}
path = "./dataset/face_data"
# path = './dataset/emotional_data'
index_label = 0
for forder_name in listdir(path):
    label_to_index[forder_name] = index_label
    index_label += 1

print(label_to_index)
# label_to_index = {'Tien_dung': 0, 'van_long': 1, 'ribi_sachi': 2, 'trong_nghia': 3, 'bich_lan': 4, 'Mai_ly': 5, 'nguyen': 6, 'thai_vu': 7, 'tran_thanh': 8, 'manh_hung': 9, 'huynh_phuong': 10, 'tien_thang': 11, 'duc_anh': 12, 'Minh_hoang': 13, 'hoai_linh': 14}


#pre-preocessing data
y_data = []
x_data = []
for forder_name in listdir(path):
    for file_name in listdir(path + '/' + forder_name):
        path_image = path + "/" + forder_name + "/" + file_name
        image = cv2.imread(path_image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        image = cv2.resize(image, (164,164))
        image = image/255
        x_data.append(image)
        y_data.append(label_to_index[forder_name])
x_data = np.array(x_data)
y_data = np.array(y_data)

# create train data and test data
X_train, X_test, y_train, y_test = train_test_split(x_data,y_data, test_size = 0.2)

print(X_train.shape,X_test.shape)
from sklearn.preprocessing import LabelBinarizer
lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.fit_transform(y_test)




batch_size = 32
epochs = 6
num_classes = 15    # face recognition
# num_classes = 5     # emotion recognition

print(datetime.datetime.now())
model_cnn = Sequential()
model_cnn.add(Conv2D(32, (5,5), activation = 'relu'))
model_cnn.add(Conv2D(32, (5,5), activation = 'relu'))
model_cnn.add(Dropout(0.15))
model_cnn.add(Conv2D(64, (5,5), activation = 'relu'))
model_cnn.add(MaxPooling2D(pool_size= (2,2)))
model_cnn.add(Conv2D(64, (5,5), activation = 'relu'))
model_cnn.add(Dropout(0.15))
# model_cnn.add(Conv2D(64, (5,5), activation = 'relu'))
model_cnn.add(Conv2D(128, (5,5), activation = 'relu'))
model_cnn.add(MaxPooling2D(pool_size=(2,2)))
model_cnn.add(Conv2D(128, (3,3), activation = 'relu'))
model_cnn.add(MaxPooling2D(pool_size=(2,2)))
model_cnn.add(Conv2D(128, (3,3), activation = 'relu'))
model_cnn.add(Dropout(0.15))
model_cnn.add(Conv2D(128, (3,3), activation = 'relu'))
# model_cnn.add(Conv2D(128, (3,3), activation = 'relu'))
model_cnn.add(MaxPooling2D(pool_size=(2,2)))
model_cnn.add(Conv2D(128, (3,3), activation = 'relu'))

model_cnn.add(Flatten())
model_cnn.add(Dense(128, activation="sigmoid"))
model_cnn.add(Dropout(0.3))
model_cnn.add(Dense(128, activation='relu'))
model_cnn.add(Dense(num_classes, activation='softmax'))


model_cnn.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])

model_cnn.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1)
# model_cnn.summary()

print(datetime.datetime.now())

# save model face recognition
model_cnn.save('./keras_cnn_model_file/model_keras1.h5')
model_cnn.save_weights('./keras_cnn_model_file//model_keras_weights1.h5')
print(model_cnn.evaluate(X_test,y_test))

