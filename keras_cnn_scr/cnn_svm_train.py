import numpy
import numpy as np
import cv2
from os import listdir
from keras.models import load_model
from keras.models import Model
from sklearn.model_selection import train_test_split
import joblib

#load label names
path = "./dataset/face_data"
cnt = 0
label_to_index = {}
for forder_name in listdir(path):
    label_to_index[forder_name] = cnt
    cnt += 1
print(label_to_index)
# label_to_index = {'Tien_dung': 0, 'van_long': 1, 'ribi_sachi': 2, 'trong_nghia': 3, 'bich_lan': 4, 'Mai_ly': 5, 'nguyen': 6, 'thai_vu': 7, 'tran_thanh': 8, 'manh_hung': 9, 'huynh_phuong': 10, 'tien_thang': 11, 'duc_anh': 12, 'Minh_hoang': 13, 'hoai_linh': 14}


y_labels = []
x_data = []
for forder_name in listdir(path):
    for file_name in listdir(path + "/" + forder_name):
        pre-processing data
        image = cv2.imread(path + "/" + forder_name +"/" + file_name)
        image = (image/255)           
        image = cv2.resize(image, (164, 164))
        y_labels.append(label_to_index[forder_name])
        x_data.append(image)

# prepare data
x_data = np.array(x_data)
y_labels = np.array(y_labels)
X_train, X_test, y_train, y_test = train_test_split(x_data, y_labels, test_size=0.05)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

#load the trained model to classify sign
model_cnn = load_model("./keras_cnn_model_file/model_keras1.h5")
model_cnn.load_weights("./keras_cnn_model_file/model_keras_weights1.h5")
model_cnn.summary()
#dictionary to label all traffic signs class.
print(label_names)



layer_name = 'flatten_1'
layer_dict = dict([(layer.name, layer) for layer in model_cnn.layers])
model_feature = Model(inputs=model_cnn.inputs, outputs=layer_dict[layer_name].output)
model_feature.summary()



x_feature_train = model_feature.predict(X_train)
x_feature_test = model_feature.predict(X_test)
print(x_feature_train.shape)

from sklearn import svm

#Create a svm Classifier
clf = svm.SVC(kernel='linear') # Linear Kernel

#Train the model using the training sets
clf.fit(x_feature_train, y_train)

filename = './keras_cnn_model_file/svm_model_keras1.sav'
joblib.dump(clf, filename)


y_pred = clf.predict(x_feature_test)
print(y_pred.shape)
from sklearn import metrics

# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))