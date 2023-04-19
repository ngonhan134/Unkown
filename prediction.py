from ROI import *
import os
import LMTRP
import joblib
import numpy as np
import cv2

# đường dẫn tới thư mục chứa các ảnh
path_out_img = './ROI1'
# Xóa toàn bộ tệp tin ảnh trong thư mục path_out_img

for file_name in os.listdir(path_out_img):
    if file_name.endswith('.bmp'):
        os.remove(os.path.join(path_out_img, file_name))


roiImageFromHand(path_out_img, option=2, cap=cv2.VideoCapture(0))
# lấy danh sách các tệp tin trong folder
file_list = os.listdir(path_out_img)

# lọc ra danh sách các ảnh trong folder
image_list = [os.path.join(path_out_img, file) for file in file_list if file.endswith('.bmp')]

# load mô hình đã được train
recognizer = joblib.load('./data/classifiers/nhan_classifier.joblib')
pred = 0

results = []
confidence_scores = []
for img in image_list:
    feature = LMTRP.LMTRP_process(cv2.imread(img))
    # feature = LMTRP.LMTRP_process(img)
    feature = feature.reshape(1, -1)
    decision = recognizer.decision_function(feature)
    confidence = 1 / (1 + np.exp(-decision))
    predict = recognizer.predict(feature)
    # print(confidence)
    if predict[0]==1:
        pred = pred + 1
        confidence_scores.append(confidence)
        # text = "Nhan"
        # print(text)

    # results.append(pred,confidence)


sum=np.sum(confidence_scores)
if pred>=5 and sum >=5: 
    print('User')
    # print(pred)
    # print(np.sum(confidence_scores))
else :
    print('Unknown')
    # print(pred)
    # print(np.sum(confidence_scores))


