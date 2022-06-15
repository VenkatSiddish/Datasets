import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.feature import hog
from skimage import exposure
from skimage.io import imread, imshow
import pickle
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image

from sklearn.decomposition import PCA

dir= '/home/furioussavenger/Data_1125'
categories=['Cloud','Rain','Sunrise','Foggy']

pick_in=open('resnet18_1125.pickle','rb')
data1=pickle.load(pick_in)
pick_in.close()

pick_in=open('hog_1125.pickle','rb')
data2=pickle.load(pick_in)
pick_in.close()

features_hog=[]
features_resnet=[]
labels_hog=[]
labels_resnet=[]
for feature,label in data1:
	features_resnet.append(feature)
	labels_resnet.append(label)
	
for feature,label in data2:
	features_hog.append(feature)
	labels_hog.append(label)
	
print(len(features_resnet[0]))
print(len(features_hog[0]))
print(len(labels_hog), len(labels_resnet))

pca=PCA(n_components=0.95)
px1=pca.fit_transform(features_resnet)
px2=pca.fit_transform(features_hog)
#print(px1.shape,px2.shape)

reduced_features=np.array([],dtype=float)
reduced_features=np.concatenate((px1,px2),axis=1)
print(reduced_features.shape)
px=pca.fit_transform(reduced_features)
#print(labels_hog)
#print(labels_resnet)
print(px.shape)
xtrain,xtest, ytrain,ytest=train_test_split(px,labels_resnet,test_size=0.25)

from sklearn.svm import SVC
model=SVC(C=1,kernel='linear',gamma='auto')
model.fit(xtrain,ytrain)
prediction=model.predict(xtest)
acurracy=model.score(xtest,ytest)

print('Accuracy',acurracy)
print('Prediction is: ',categories[prediction[0]])
cm=confusion_matrix(ytest,prediction)
print(cm)

from sklearn.model_selection import cross_val_score,cross_val_predict
clf = SVC(kernel='linear', C=1, random_state=42)
scores = cross_val_score(clf, px, labels_resnet, cv=5)
print(scores)
prediction=cross_val_predict(clf,px,labels_resnet)
cm=confusion_matrix(labels_resnet,prediction)
print(cm)


'''
import xgboost as xgb
from sklearn.metrics import mean_squared_error

data_dmatrix = xgb.DMatrix(data=px,label=labels_resnet)
xg_reg = xgb.XGBClassifier(colsample_bytree = 0.3, learning_rate = 0.1,max_depth = 20, alpha = 10, n_estimators = 10)
xg_reg.fit(xtrain,ytrain)
accuracy=xg_reg.score(xtest,ytest)
print(accuracy)
preds = xg_reg.predict(xtest)
cm=confusion_matrix(ytest,preds)
print(cm)
'''

