import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import normalize

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
from scipy.stats import entropy
dir= '/home/ameeth/WC'

categories=['Cloud','Rain','Shine','Sunrise','Foggy']
pick_in=open('data5_deep.pickle','rb')
data1=pickle.load(pick_in)
pick_in.close()

random.shuffle(data1)
features=[]
labels=[]

for feature,label in data1:
	features.append(feature)
	labels.append(label)

print(len(features[0]))

new=pd.DataFrame(features)
print(type(new))
norm=normalize(new)
new_scaled=new.var()

print(new_scaled.describe())
max_el=new_scaled.max()
print(max_el)
#print(new_scaled.max)
	
from sklearn.feature_selection import VarianceThreshold

selector=VarianceThreshold(threshold=0.082)
X_train_vrth=selector.fit_transform(features)
n_features1=X_train_vrth.shape[1]
print('Features after variance threshold %d with full FV',n_features1)

#X_train_vr = selector.transform(pxtrain)
#X_test_vr  =  selector.transform(pxtest)
x_vr=selector.transform(features)
print(x_vr.shape)
#print(X_train_vr.shape)
#print(X_test_vr.shape)


print(type(x_vr))
px_vr=np.array(x_vr,dtype=float)

from sklearn.decomposition import PCA
pca=PCA(n_components=0.95)
px=pca.fit_transform(px_vr)
pxtrain,pxtest, pytrain,pytest=train_test_split(px,labels,test_size=0.25)	
#xtrain,xtest, ytrain,ytest=train_test_split(features,labels,test_size=0.25,random_state=42)

x_train=np.array(pxtrain,dtype=float)

def shan_entropy(c):
    c_normalized = c / float(np.sum(c))
    c_normalized = c_normalized[np.nonzero(c_normalized)]
    H = -sum(c_normalized* np.log(c_normalized))  
    return H


def calc_MI(X,Y,bins):

   c_XY = np.histogram2d(X,Y,bins)[0]
   c_X = np.histogram(X,bins)[0]
   c_Y = np.histogram(Y,bins)[0]

   H_X = shan_entropy(c_X)
   H_Y = shan_entropy(c_Y)
   H_XY = shan_entropy(c_XY)

   MI = H_X + H_Y - H_XY
   return MI


r,c =x_train.shape
print(r,c)
x_test=np.array(pxtest,dtype=float)
y_test=np.array(pytest,dtype=float)
y_train=np.array(pytrain,dtype=float)

'''
ent = np.empty(c)

maxc=0;
sumc=0;
minc=10;
for i in range (c):
    ent[i]=entropy(x_train[:,i])
    if(ent[i]>maxc):
    	maxc=ent[i]
    elif(ent[i]<minc):
    	minc=ent[i]
    sumc+=ent[i]
print("Entropy of data")
print(ent)
print(maxc)
print(minc)
print(sumc/c)
thresh=np.arange(6,6.4)
print(thresh)
ent_slct = np.where(ent > 6)[0]
ent_slct1 = np.where(ent > 6.4)[0]
print(ent_slct)
print(ent_slct1)

x_train_ent= x_train[:,ent_slct]
print(x_train_ent.shape[0])
print(x_train_ent.shape[1])
x_test_ent =x_test[:,ent_slct]
'''



from sklearn.svm import SVC
model=SVC(C=1,kernel='linear',gamma='auto')
model.fit(x_train,y_train)
prediction=model.predict(x_test)
acurracy=model.score(x_test,y_test)

print('Accuracy',acurracy)

#print('Prediction is: ',categories[prediction[0]])
cm=confusion_matrix(y_test,prediction)
print(cm)
#print('Accuracy entropy with full fv [threshold >5.9] SVC: {}%'.format(acurracy * 100))

'''
x_train_ent1= x_train[:,ent_slct1]
print(x_train_ent1.shape[0])
print(x_train_ent1.shape[1])
x_test_ent1 =x_test[:,ent_slct1]
from sklearn.svm import SVC
model=SVC(C=1,kernel='linear',gamma='auto')
model.fit(x_train_ent1,y_train)
prediction=model.predict(x_test_ent1)
acurracy1=model.score(x_test_ent1,y_test)


#print('Accuracy',acurracy)
categories=['Cloud','Rain','Shine','Sunrise']
#print('Prediction is: ',categories[prediction[0]])
cm=confusion_matrix(y_test,prediction)
print(cm)
print('Accuracy entropy with full fv [threshold >6.2] SVC: {}%'.format(acurracy1 * 100))
'''

'''
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier()
model.fit(x_train,y_train)
prediction=model.predict(x_test)
acurracy=model.score(x_test,y_test)

#print(accuracy)


#print('Prediction is: ',categories[forest_predictions[0]])

#myimage=xtest[0].reshape(63,60)
#plt.imshow(myimage)
#plt.show()

# creating a confusion matrix
cm = confusion_matrix(y_test, prediction)
print(cm)
print('Accuracy entropy with full fv [threshold >5.9] RFC: {}%'.format(acurracy * 100))
'''
'''
x_train_ent1= x_train[:,ent_slct1]
print(x_train_ent1.shape[0])
print(x_train_ent1.shape[1])
x_test_ent1 =x_test[:,ent_slct1]
model.fit(x_train_ent1,y_train)
prediction=model.predict(x_test_ent1)
acurracy1=model.score(x_test_ent1,y_test)

#print('Accuracy',acurracy)
categories=['Cloud','Rain','Shine','Sunrise']
#print('Prediction is: ',categories[prediction[0]])
cm=confusion_matrix(y_test,prediction)
print(cm)
print('Accuracy entropy with full fv [threshold >6.2] RFC: {}%'.format(acurracy1 * 100))
'''

'''
x_train_ent= x_train_sample[:,ent_slct]
print(x_train_ent.shape[0])
print(x_train_ent.shape[1])
x_test_ent =x_test_sample[:,ent_slct]

from sklearn.svm import SVC
model=SVC(C=1,kernel='linear',gamma='auto')
model.fit(x_train_ent,y_train)
prediction=model.predict(x_test_ent)
acurracy=model.score(x_test_ent,y_test)

#print('Accuracy',acurracy)
categories=['Cloud','Rain','Shine','Sunrise']
#print('Prediction is: ',categories[prediction[0]])
cm=confusion_matrix(y_test,prediction)
print(cm)
print('Accuracy entropy with full fv [threshold >5.9] SVC: {}%'.format(acurracy * 100))

'''

'''
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

bins = 6 # 
n = x_train.shape[1]
matMI = np.zeros((n, n))
print(n)
for ix in np.arange(n):
    for jx in np.arange(ix+1,n):
    	if(calc_MI(x_train[:,ix], x_train[:,jx], bins)>0):
        	matMI[ix,jx] = calc_MI(x_train[:,ix], x_train[:,jx], bins)
print("Mutual information between vectors")
#print(matMI)

mi = mutual_info_classif(x_train,y_train)
#print(mi.shape[])
#mi1=np.array([])
mi1=[]
i=0
nxtrain=[[]]
for j in range(0,mi.shape[0]):
	if(mi[j]>0):
		#mi1[i]=mi[j]
		#i=i+1
		mi1.append(mi[j])
		#x_train.delete([:j])
		nxtrain.append(x_train[:,j])
n_xtrain=np.array(nxtrain,dtype=object)
print(n_xtrain.shape)
print(x_train.shape)
nmi1=np.array(mi1,dtype=float)
mi_1=np.sort(nmi1)
'''

'''
print("MI")
print(mi_1)
mi_1 = pd.Series(mi_1)
df = pd.DataFrame(x_train)
mi_1.index =df.columns
plt.plot(mi)
plt.show()
mi_1.sort_values(ascending=False).plot.bar(figsize=(20, 6))
plt.title("MI vs Feature No")
plt.ylabel('Mutual Information')
plt.xlabel('Feature No')
plt.show()
'''
