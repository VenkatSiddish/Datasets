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

from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.preprocessing import label_binarize
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import roc_auc_score,plot_roc_curve,auc

def plot_sklearn_roc_curve(y_real, y_pred):
    fpr, tpr, _ = roc_curve(y_real, y_pred)
    roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr)
    #roc_display.figure_.set_size_inches(5,5)
    #plt.plot([0, 1], [0, 1], color = 'g')
'''
n_classes = len(set(labels))

Y = label_binarize(labels, classes=[*range(n_classes)])

xtrain, xtest, ytrain, ytest = train_test_split(norm,Y,random_state = 42)

clf = OneVsRestClassifier(SVC(C=1,kernel='linear',gamma='auto',probability=True,random_state=42))

clf.fit(xtrain, ytrain)

y_score = clf.predict_proba(xtest)


# precision recall curve
precision = dict()
recall = dict()
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(ytest[:, i],
                                                        y_score[:, i])
    plt.plot(recall[i], precision[i], lw=2, label='class {}'.format(i))
    
plt.xlabel("recall")
plt.ylabel("precision")
plt.legend(categories,loc="best")
plt.title("precision vs. recall curve")
plt.show()
'''

xtrain,xtest, ytrain,ytest=train_test_split(norm,labels,test_size=0.25,random_state=42)
model=SVC(C=1,kernel='linear',gamma='auto',probability=True)
model.fit(xtrain,ytrain)
y_proba=model.predict_proba(xtest)
'''
import scikitplot as skplt
import matplotlib.pyplot as plt

skplt.metrics.plot_roc(ytest, y_proba)
plt.show()
'''
# Plots the Probability Distributions and the ROC Curves One vs Rest
plt.figure(figsize = (12, 8))
bins = [i/20 for i in range(20)] + [1]
classes = model.classes_
roc_auc_ovr = {}
for i in range(len(classes)):
    # Gets the class
    c = classes[i]
    
    # Prepares an auxiliar dataframe to help with the plots
    df_aux = pd.DataFrame(xtest.copy())
    df_aux['class'] = [1 if y == c else 0 for y in ytest]
    df_aux['prob'] = y_proba[:, i]
    df_aux = df_aux.reset_index(drop = True)
    print(df_aux.head())
    exit()
    '''
    # Plots the probability distribution for the class and the rest
    ax = plt.subplot(2, 4, i+1)
    sns.histplot(x = "prob", data = df_aux, hue = 'class', color = 'b', ax = ax, bins = bins)
    ax.set_title(c)
    ax.legend([f"Class: {categories[c]}", "Rest"])
    ax.set_xlabel(f"P(x = {c})")
    '''
    # Calculates the ROC Coordinates and plots the ROC Curves
    #ax_bottom = plt.subplot(2, 4, i+1)
    tpr, fpr, _ = roc_curve(df_aux['class'], df_aux['prob'])
    #print(tpr.shape,fpr.shape)
    auc_score=auc(fpr,tpr)
    roc_display=RocCurveDisplay(fpr=fpr, tpr=tpr,roc_auc=auc_score,estimator_name=categories[c]).plot()
    #ax_bottom.set_xlabel('False Positive Rate')
    #ax_bottom.set_ylabel('True Positive Rate')
    #ax_bottom.set_title("ROC Curve OvR")
    roc_display.figure_.set_size_inches(5,5)
    plt.plot([0, 1], [0, 1], color = 'g')
    
    # Calculates the ROC AUC OvR
    roc_auc_ovr[c] = roc_auc_score(df_aux['class'], df_aux['prob'])
plt.tight_layout()
plt.show()

# Displays the ROC AUC for each class
avg_roc_auc = 0
i = 0
for k in roc_auc_ovr:
    avg_roc_auc += roc_auc_ovr[k]
    i += 1
    print(f"{k} ROC AUC OvR: {roc_auc_ovr[k]:.4f}")
print(f"average ROC AUC OvR: {avg_roc_auc/i:.4f}")


