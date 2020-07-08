# from keras.models import load_model
from os import listdir
import numpy as np
# import keras
from keras.models import load_model
# from tensorflow import keras
from mtcnn.mtcnn import MTCNN
import cv2

index_to_labes = {}
# label_to_index = {'Tien_dung': 0, 'van_long': 1, 'ribi_sachi': 2, 'trong_nghia': 3, 'bich_lan': 4, 'Mai_ly': 5, 'nguyen': 6, 'thai_vu': 7, 'tran_thanh': 8, 'manh_hung': 9, 'huynh_phuong': 10, 'tien_thang': 11, 'duc_anh': 12, 'Minh_hoang': 13, 'hoai_linh': 14}
label_to_index = {'manh_hung': 0, 'Mai_ly': 1, 'van_long': 2, 'Tien_dung': 3, 'bich_lan': 4, 'nguyen': 5, 'trong_nghia': 6, 'tien_thang': 7, 'Truong Thanh Sang_1712945': 8, 'Phung Tuan Hung_1611444': 9, 'Nguyen Thi Tuyet_1613947': 10, 'Truong Minh Tuan_1613935': 11, 'Dinh Manh Cuong_1510352': 12, 'Dao Duy Hanh_1711205': 13, 'Hoang Vu Nam_1512062': 14, 'Luu Anh Khoa_1611610': 15, 'Huynh Vuong Vu_1514097': 16, 'Le Phuc Thanh_1613178': 17, 'Dinh Giang Nam_1512059': 18, 'Nguyen Phu Cuong_1710725': 19, 'Ngo Trong Huu_1511438': 20, 'duc_thinh': 21, 'duy_quang': 22, 'hoang_hiep': 23, 'ngoc_anh': 24, 'phuoc_loc': 25, 'hieu_nguyen': 26, 'xuan_tien': 27}

# label_to_index = {}
# path = "/home/vanlong/vanlong/ky6/doAn/data/raw"
# index_label = 0
# for forder_name in listdir(path):
    # label_to_index[forder_name] = index_label
    # index_label += 1
# print(label_to_index)


for key,value in label_to_index.items():
    index_to_labes[value] = key
print(index_to_labes)


#Load model
# loaded_model = load_model('./keras_cnn_model_file/model_keras1.h5')
# loaded_model.load_weights('./keras_cnn_model_file/model_keras_weights1.h5')
loaded_model = load_model('./keras_cnn_model_file/model_keras_face.h5')
loaded_model.load_weights('./keras_cnn_model_file/model_keras_weights_face.h5')
# loaded_model.summary()  
print(loaded_model)
detector = MTCNN()
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret,frame = cap.read()
    # print(frame.shape)
    frame = cv2.resize(frame,(640,int(frame.shape[0]/frame.shape[1]*640)))  
    if ret:
        faces = detector.detect_faces(frame)
        print(len(faces))
        for person in faces:
            bounding_box = person['box']
            im_crop = frame[bounding_box[1]: bounding_box[1] + bounding_box[3], bounding_box[0]: bounding_box[0]+bounding_box[2] ]
            print(bounding_box[0],bounding_box[1],bounding_box[2],bounding_box[3])
            print(im_crop.shape)
            if im_crop.shape[0] > 0 and im_crop.shape[1] > 0:
                # im_crop = cv2.cvtColor(im_crop, cv2.COLOR_BGR2HSV)
                # im_crop = cv2.resize(im_crop,(164,164))
                im_crop = cv2.resize(im_crop,(224,224))
                im_crop = (im_crop/255)   
                print(im_crop.shape)        #normalize
                im_crop = [im_crop]
                im_crop = np.array(im_crop)
                prediction = loaded_model.predict(im_crop)[0]
                for i in range(len(prediction)):
                    print(index_to_labes[i],':',prediction[i])
                max_index = int(np.argmax(prediction))
                cv2.rectangle(frame,(bounding_box[0], bounding_box[1]),(bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),(0,155,255),2)
                print('\n',index_to_labes[max_index],prediction[max_index])
                if prediction[max_index] > 0.3:
                    cv2.putText(frame, index_to_labes[max_index] + " " + str(np.round(prediction[max_index],2)) , (bounding_box[0], bounding_box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (30, 255, 30), 2, cv2.LINE_AA)
        cv2.imshow("frame",frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
              break
    else:
        break
cap.release()
cv2.destroyAllWindows()

