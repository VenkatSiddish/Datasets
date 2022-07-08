#Imports
import os
import numpy as np
import pandas as pd
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
from sklearn.preprocessing import normalize,StandardScaler

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
import seaborn as sns

#Directory and Categories for classification
#dir= '/home/ameeth/WC/'

categories=['Cloudy','Rain','Shine','Sunrise']
target_names = ['Cloudy','Rain','Shine','Sunrise']

#Opening feature picle files

pick_in=open('denseNet161_1125.pickle','rb')
data1=pickle.load(pick_in)
pick_in.close()

pick_in=open('hog_1125.pickle','rb')
data2=pickle.load(pick_in)
pick_in.close()

#random.shuffle(data1)
#random.shuffle(data2)

#Splitting Data into features and labels and normalizing data
features_hog=[]
features=[]
labels_new=[]
#labels_hog=[]
for feature,label in data1:
	features.append(feature)
	labels_new.append(label)

for feature,label in data2:
	features_hog.append(feature)
	#labels_hog.append(label)



print(len(features[0]))
print(len(labels_new))

df=pd.DataFrame(features)
nor=normalize(df)
norm1=pd.DataFrame(nor)
#Removing zero value columns
norm1 = norm1.loc[:, (norm1 != 0).any(axis=0)]

norm2=normalize(features_hog)

#Feature Selection Using Mututal Information

from sklearn.feature_selection import SelectPercentile as SP
selector1 = SP(percentile=50) # select features with top 50% MI scores

selector1.fit(norm1,labels_new)
X_4_1 = selector1.transform(norm1)
print(X_4_1.shape, type(X_4_1))
selector2 = SP(percentile=50) # select features with top 50% MI scores
selector2.fit(norm2,labels_new)
X_4_2 = selector2.transform(norm2)
print(X_4_2.shape, type(X_4_2))

#normed=np.concatenate((X_4_1,X_4_2),axis=1)
normed=np.concatenate((norm1,norm2),axis=1)

df_new=pd.DataFrame(normed)
df_new['labels']=pd.DataFrame(labels_new)
#print(df_new.head())
shuffled = df_new.sample(frac=1,random_state=42).reset_index()
shuffled.drop(shuffled.columns[0],axis=1,inplace=True)
#print(shuffled.head())
labels=shuffled.iloc[:,-1]
#print(labels.head())
norm=shuffled.drop(shuffled.columns[-1],axis=1)
#print(norm.head())

from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
from itertools import cycle

y = label_binarize(labels, classes=[0, 1, 2, 3])
n_classes = y.shape[1]
print(n_classes)


xtrain,xtest, ytrain,ytest=train_test_split(norm,y,test_size=0.25,random_state=42)

classifier = OneVsRestClassifier(SVC(kernel="linear", probability=True, random_state=42))
y_score = classifier.fit(xtrain, ytrain).decision_function(xtest)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(ytest[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(ytest.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
for i in range(n_classes):
	ax=plt.subplot(1,4,i+1)
	lw = 2
	plt.plot(
	    fpr[i],
	    tpr[i],
	    color="darkorange",
	    lw=lw,
	    label="ROC curve (area = %0.2f)" % roc_auc[i],
	)
	plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	ax.set_xlabel("False Positive Rate")
	ax.set_ylabel("True Positive Rate")
	ax.set_title([f"{categories[i]}", "vs Rest"])
	ax.legend(loc="lower right")
	plt.suptitle("Receiver operating characteristic for each class")
plt.show()


