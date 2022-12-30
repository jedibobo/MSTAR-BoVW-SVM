import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import sklearn

from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split

def normalize_and_resize_image(image, gamma):
    # 将图像转换为浮点型
    image = image.astype(np.float32)
    # 计算归一化后的像素值
    normalized_image = image**(1/gamma)
    # 将像素值转换为0到255的范围
    normalized_image = (normalized_image * 255).astype(np.uint8)
    # 将图像缩放到128x128
    normalized_image = cv2.resize(normalized_image, (128, 128))
    return normalized_image

def read_data(label2id):
    X = []
    Y = []
    for label in os.listdir('SARimage/ALL'):
        for img_file in os.listdir(os.path.join('SARimage/ALL', label)):
            img = cv2.imread(os.path.join('SARimage/ALL', label, img_file))
            img = normalize_and_resize_image(img, 2)
            X.append(img)
            Y.append(label2id[label])
    return X, Y


# Label to id, used to convert string label to integer 
label2id = {'2S1':0, 'BMP2(SN_9566)':1, 'BRDM_2':2, 'BTR70(SN-C71)':3, 'BTR_60':4,'D7':5,'T62':6,'T72(SN_132)':7,'ZIL131':8,'ZSU_23_4':9}
X, Y = read_data(label2id)


def extract_sift_features(X):
    image_descriptors = []
    sift = cv2.xfeatures2d.SIFT_create()

    for i in range(len(X)):
        kp, des = sift.detectAndCompute(X[i], None)
        image_descriptors.append(des)

    return image_descriptors

image_descriptors = extract_sift_features(X)


all_descriptors = []
for descriptors in image_descriptors:
    if descriptors is not None:
        for des in descriptors:
            all_descriptors.append(des)

def kmeans_bow(all_descriptors, num_clusters):
    bow_dict = []
    kmeans = KMeans(n_clusters=num_clusters).fit(all_descriptors)
    bow_dict = kmeans.cluster_centers_
    return bow_dict

num_clusters = 100

if not os.path.isfile('SARimage/bow_dictionary100.pkl'):
    BoW = kmeans_bow(all_descriptors, num_clusters)
    pickle.dump(BoW, open('SARimage/bow_dictionary100.pkl', 'wb'))
else:
    BoW = pickle.load(open('SARimage/bow_dictionary100.pkl', 'rb'))


#
def create_features_bow(image_descriptors, BoW, num_clusters):
    X_features = []
    for i in range(len(image_descriptors)):
        features = np.array([0] * num_clusters)

        if image_descriptors[i] is not None:
            distance = cdist(image_descriptors[i], BoW)
            argmin = np.argmin(distance, axis=1)
            for j in argmin:
                features[j] += 1
        X_features.append(features)
    return X_features

X_features = create_features_bow(image_descriptors, BoW, num_clusters)


#
X_train = [] 
X_test = []
Y_train = []
Y_test = []
X_train, X_test, Y_train, Y_test = train_test_split(X_features, Y, test_size=0.2, random_state=42)

svm = sklearn.svm.SVC(C=1.0, gamma='auto', probability=True)
svm.fit(X_train, Y_train)

#Thu predict 
img_test = cv2.imread('SARimage\TEST\\2S1\HB14931.jpeg')
img = [img_test]
img_sift_feature = extract_sift_features(img)
img_bow_feature = create_features_bow(img_sift_feature, BoW, num_clusters)
img_predict = svm.predict(img_bow_feature)

print(img_predict)
for key, value in label2id.items():
    if value == img_predict[0]:
        print('Your prediction: ', key)

#Accuracy
print(svm.score(X_test, Y_test))

#Show image
cv2.imshow("Img", img_test)
cv2.waitKey()