# import cv2
# import numpy as np
# import os
# from LMTRP import LMTRP_process

# def extract_features(image_path):
#     image = cv2.imread(image_path)
#     feature = LMTRP_process(image)
#     return feature
# def load_train_data(train_folder_path):
#     X = []
#     y = []
#     for person_folder in os.listdir(train_folder_path):
#         person_folder_path = os.path.join(train_folder_path, person_folder)
#         for image_name in os.listdir(person_folder_path):
#             image_path = os.path.join(person_folder_path, image_name)
#             feature = extract_features(image_path)
#             X.append(feature)
#             y.append(person_folder)
#     return X, y
# def load_test_data(test_folder_path):
#     X_test = []
#     y_test = []
#     for person_folder in os.listdir(test_folder_path):
#         person_folder_path = os.path.join(test_folder_path, person_folder)
#         for image_name in os.listdir(person_folder_path):
#             image_path = os.path.join(person_folder_path, image_name)
#             feature = extract_features(image_path)
#             X_test.append(feature)
#             y_test.append(person_folder)
#     return X_test, y_test
# def compute_distances(X_train, X_test):
#     distances = []
#     for feature_test in X_test:
#         distance = []
#         for feature_train in X_train:
#             dist = np.linalg.norm(feature_test - feature_train)
#             distance.append(dist)
#         distances.append(distance)
#     return np.array(distances)
# def predict(X_train, y_train, X_test, y_test):
#     distances = compute_distances(X_train, X_test)
#     predictions = []
#     for i in range(len(X_test)):
#         nearest = np.argmin(distances[i])
#         if y_train[nearest] == y_test[i]:
#             predictions.append(1)
#         else:
#             predictions.append(0)
#     accuracy = np.mean(predictions)
#     print("Accuracy:", accuracy)

# # Load train data
# X_train, y_train = load_train_data('./DATA3/train/')

# # Load test data
# X_test, y_test =load_test_data('./DATA3/test/')

# predict(X_train, y_train, X_test, y_test)


# import os
# import cv2
# import numpy as np
# from sklearn.model_selection import train_test_split

# from LMTRP import LMTRP_process

# # Đường dẫn tới thư mục chứa dữ liệu
# data_dir = "./data3/train/"

# # Khai báo các biến
# X = []
# y = []
# X_test = []
# y_test = []

# # Duyệt qua từng thư mục trong data_dir để tạo dữ liệu
# for person_folder in os.listdir(data_dir):
#     person_folder_path = os.path.join(data_dir, person_folder)
#     # Duyệt qua từng ảnh trong thư mục của từng người
#     for image_name in os.listdir(person_folder_path):
#         image_path = os.path.join(person_folder_path, image_name)
#         image = cv2.imread(image_path)
#         print("Processing image:", image_path)
#         feature = LMTRP_process(image)
#         X.append(feature)
#         y.append(person_folder)

# # Chia dữ liệu thành tập train và tập test (80% train, 20% test)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# # Tính toán khoảng cách Euclid giữa các vector đặc trưng trong tập train và tập test
# distances = []
# for feature_test in X_test:
#     distance = []
#     for feature_train in X_train:
#         dist = np.sqrt(np.sum(np.square(feature_test - feature_train)))
#         distance.append(dist)
#     distances.append(distance)
# distances = np.array(distances)

# # Dự đoán kết quả và tính độ chính xác
# predictions = []
# for i in range(len(X_test)):
#     nearest = np.argmin(distances[i])
#     if y_train[nearest] == y_test[i]:
#         predictions.append(1)
#     else:
#         predictions.append(0)
# accuracy = np.mean(predictions)
# print("Accuracy:", accuracy)

# import os
# import cv2
# import numpy as np
# from sklearn.model_selection import train_test_split

# from LMTRP import LMTRP_process

# # Đường dẫn tới thư mục chứa dữ liệu
# data_dir = "./data1/"

# # Khai báo các biến
# X = []
# y = []
# X_test = []
# y_test = []

# # Duyệt qua từng thư mục trong data_dir để tạo dữ liệu
# for person_folder in os.listdir(data_dir):
#     person_folder_path = os.path.join(data_dir, person_folder)
#     # Duyệt qua từng ảnh trong thư mục của từng người
#     for image_name in os.listdir(person_folder_path):
#         image_path = os.path.join(person_folder_path, image_name)
#         image = cv2.imread(image_path)
#         print("Processing image:", image_path)
#         feature = LMTRP_process(image)
#         X.append(feature)
#         y.append(person_folder)

# # Chia dữ liệu thành tập train và tập test (80% train, 20% test)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Tính toán điểm tương tự giữa các vector đặc trưng trong tập train và tập test
# scores = []
# for feature_test in X_test:
#     score = []
#     for feature_train in X_train:
#         dist = np.sum(np.square(feature_test - feature_train)) / len(feature_test)
#         score.append(dist)
#     scores.append(score)
# scores = np.array(scores)

# # Dự đoán kết quả và tính độ chính xác
# predictions = []
# for i in range(len(X_test)):
#     nearest = np.argmin(scores[i])
#     if y_train[nearest] == y_test[i]:
#         predictions.append(1)
#     else:
#         predictions.append(0)
# accuracy = np.mean(predictions)
# print("Accuracy:", accuracy)
# import os
# import numpy as np
# import cv2
# from LMTRP import LMTRP_process

# from tqdm import tqdm
# import pandas as pd

# from scipy.spatial.distance import cdist
# from sklearn.model_selection import train_test_split


# # Set up dataset
# data_dir = "./data/Nhan/"
# # data_folder = "./DATA3/train/"
# num_people = 13
# num_images_per_person = 10

# def extract_features(image_path):
#     try:
#         image = cv2.imread(image_path)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         feature = LMTRP_process(image)
#         return feature
#     except Exception as e:
#         print(f"Error extracting features from {image_path}: {e}")
#         return None


# # Load images and extract features
# X = []
# y = []
# for person_folder in os.listdir(data_dir):
#     person_folder_path = os.path.join(data_dir, person_folder)
#     # Duyệt qua từng ảnh trong thư mục của từng người
#     for image_name in os.listdir(person_folder_path):
#         image_path = os.path.join(person_folder_path, image_name)
#         image = cv2.imread(image_path)
#         print("Processing image:", image_path)
#         feature = LMTRP_process(image)
#         X.append(feature)
#         # y.append(person_folder)

# X = np.array(X)
# # y = np.array(y)
# x=np.save('feature.npy')
# y=np.save('lab.npy')
# # Split dataset into train and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Train model
# train_vectors = X_train.reshape(X_train.shape[0], -1)

# # Test model
# correct_predictions = 0
# threshold = 4.2  # Set threshold to determine if the predicted label is correct

# for i in range(len(X_test)):
#     query_feature = X_test[i]
#     query_vector = query_feature.reshape(1, -1)

#     distances = cdist(query_vector, train_vectors, 'euclidean')
#     min_index = np.argmin(distances)
#     prediction = y_train[min_index]
    
#     if distances[0][min_index] < threshold and prediction == y_test[i]:
#         correct_predictions += 1

# accuracy = correct_predictions / len(X_test)
# print("Accuracy:", accuracy)
# import os
# import numpy as np
# import cv2
# from LMTRP import LMTRP_process

# from tqdm import tqdm
# import pandas as pd

# from scipy.spatial.distance import cdist
# from sklearn.model_selection import train_test_split


# # Set up dataset
# data_dir = "./data/Nhan/"
# # data_folder = "./DATA3/train/"
# num_people = 13
# num_images_per_person = 10

# def extract_features(image_path):
#     try:
#         image = cv2.imread(image_path)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         feature = LMTRP_process(image)
#         return feature
#     except Exception as e:
#         print(f"Error extracting features from {image_path}: {e}")
#         return None


# # Load images and extract features
# X = []
# y = []
# for person_folder in os.listdir(data_dir):
#     person_folder_path = os.path.join(data_dir, person_folder)
#     # Duyệt qua từng ảnh trong thư mục của từng người
#     for image_name in os.listdir(person_folder_path):
#         image_path = os.path.join(person_folder_path, image_name)
#         image = cv2.imread(image_path)
#         print("Processing image:", image_path)
#         feature = LMTRP_process(image)
#         X.append(feature)
#         y.append(person_folder)

# X = np.array(X)
# y = np.array(y)
# np.save('features.npy', X)
# np.save('labels.npy', y)
# import numpy as np
# import os, cv2
# import LMTRP
# from sklearn import svm
# from joblib import dump

# from sklearn.model_selection import GridSearchCV

# def train_classifer(name):
#     # Read all the images in custom data-set
#     path = os.path.join(os.getcwd()+"/data/"+name+"/")

#     features = []
#     labels = []
#     num_images = 0

#     # Store images in a numpy format and corresponding labels in labels list
#     for root, dirs, files in os.walk(path):
#         for file in files:
#             imgpath = os.path.join(root, file)
#             img = cv2.imread(imgpath)
#             feature = LMTRP.LMTRP_process(img) # extract feature from image
#             features.append(feature)
#             num_images += 1
#             print("Number of images with features extracted:", num_images)

#             if name in file:
#                 labels.append(1) # if image belongs to name, label as 1 (true)
#             else:
#                 labels.append(-1) # if image doesn't belong to name, label as -1 (false)

#     features = np.asarray(features)
#     # features = features.reshape(features.shape[0],-1)
#     np.save('features.npy', feature)
# train_classifer("ROI")
import os
import numpy as np
import cv2
from LMTRP import LMTRP_process

# Set directory and file name to save numpy array
data_dir = './data/Nhan'
save_file = 'features.npy'
def extract_features(image_path):
    image = cv2.imread(image_path)
    feature = LMTRP_process(image)          
    return feature
num_images = 0
# Set up list to store features
features = []

# Loop through images in directory and extract features
for filename in os.listdir(data_dir):
    if filename.endswith('.bmp') or filename.endswith('.png'):
        image_path = os.path.join(data_dir, filename)
        num_images += 1
        print("Number of images with features extracted:", num_images)
        feature = extract_features(image_path)
        features.append(feature)

# Convert list to numpy array and save
features = np.array(features)
np.save(save_file, features)
