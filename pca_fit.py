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

from sklearn.decomposition import PCA

dir= '/home/ameeth/WC/'
categories=['Cloud','Rain','Sunrise','Foggy']
#categories=['HAZE','RAINY','SUNNY','SNOWY']
#categories=['Cloudy','Foggy','Sunny','Snowy','Rainy']

pick_in=open('resnet18_1125.pickle','rb')
#pick_in=open('resnet18_mwi.pickle','rb')
data1=pickle.load(pick_in)
pick_in.close()

pick_in=open('hog_1125.pickle','rb')
#pick_in=open('hog_mwi.pickle','rb')
data2=pickle.load(pick_in)
pick_in.close()

random.shuffle(data1)
random.shuffle(data2)
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
norm1=normalize(features_resnet)
norm2=normalize(features_hog)
pca=PCA(n_components=0.95)
px1=pca.fit_transform(norm1)
px2=pca.fit_transform(norm2)
print(px1.shape,px2.shape)

accuracies={}

reduced_features=np.array([],dtype=float)
reduced_features=np.concatenate((px1,px2),axis=1)
print(reduced_features.shape)
red_feat=pd.DataFrame(reduced_features)
print(red_feat.head())
#norm=normalize(reduced_features)
px=pca.fit_transform(reduced_features)
#print(labels_hog)
#print(labels_resnet)
print(px.shape)

from sklearn.feature_selection import SelectPercentile as SP
selector = SP(percentile=65) # select features with top 50% MI scores

selector.fit(reduced_features,labels_resnet)
X_4 = selector.transform(reduced_features)
xtrain,xtest,ytrain,ytest = train_test_split(
    X_4,labels_resnet
    ,random_state=42
    ,stratify=labels_resnet
)


#xtrain,xtest, ytrain,ytest=train_test_split(reduced_features,labels_resnet,test_size=0.25,random_state=42)

from sklearn.svm import SVC
model=SVC(C=1,kernel='linear',gamma='auto')
model.fit(xtrain,ytrain)
prediction=model.predict(xtest)
accuracy1=model.score(xtest,ytest)
accuracy_train1=model.score(xtrain,ytrain)

print('Training accuracy', accuracy_train1)

print('Test Accuracy',accuracy1)
print('Prediction is: ',categories[prediction[0]])
cm=confusion_matrix(ytest,prediction)
print(cm)
import seaborn as sns
group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in cm.flatten()/np.sum(cm)]
labels = [f"{v1}\n{v2}\n" for v1, v2 in zip(group_counts,group_percentages)]
n=len(categories)
labels = np.asarray(labels).reshape(n,n)
ax = sns.heatmap(cm, annot=labels, fmt='', cmap='Blues')
ax.set_title(f'Seaborn Confusion Matrix with labels for SVM with Accuracy of {accuracy1*100:.02f}%\n');
ax.set_xlabel('\nPredicted Weather Category')
ax.set_ylabel('Actual Weather Category ');
## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(categories)
ax.yaxis.set_ticklabels(categories)
## Display the visualization of the Confusion Matrix.
plt.show()

from sklearn.model_selection import cross_val_score,cross_val_predict
clf = SVC(kernel='linear', C=1, random_state=42)
scores = cross_val_score(clf, reduced_features, labels_resnet, cv=10)

print(scores)
accuracies["SVM"]=scores.max()
prediction=cross_val_predict(clf,reduced_features,labels_resnet)
cm=confusion_matrix(labels_resnet,prediction)
print(cm)
##########################################################################################################################################################################
k=6
from sklearn.ensemble import RandomForestClassifier
rfc_model=RandomForestClassifier(max_depth=k,n_estimators=500)
rfc_model.fit(xtrain,ytrain)
forest_predictions= rfc_model.predict(xtest)
accuracy2=rfc_model.score(xtest,ytest)
accuracy_train2=rfc_model.score(xtrain,ytrain)
print('Training accuracy', accuracy_train2)

print('Test Accuracy',accuracy2)

print('Prediction is: ',categories[forest_predictions[0]])

#myimage=xtest[0].reshape(63,60)
#plt.imshow(myimage)
#plt.show()

# creating a confusion matrix
cm = confusion_matrix(ytest, forest_predictions)
print(cm)

import seaborn as sns
group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in cm.flatten()/np.sum(cm)]
labels = [f"{v1}\n{v2}\n" for v1, v2 in zip(group_counts,group_percentages)]
n=len(categories)
labels = np.asarray(labels).reshape(n,n)
ax = sns.heatmap(cm, annot=labels, fmt='', cmap='Blues')
ax.set_title(f'Seaborn Confusion Matrix with labels for Random Forest Classifier with Accuracy of {accuracy2*100:.02f}%\n');
ax.set_xlabel('\nPredicted Weather Category')
ax.set_ylabel('Actual Weather Category ');
## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(categories)
ax.yaxis.set_ticklabels(categories)
## Display the visualization of the Confusion Matrix.
plt.show()

from sklearn.model_selection import cross_val_score,cross_val_predict
clf = RandomForestClassifier()
scores = cross_val_score(clf, reduced_features, labels_resnet, cv=10)

print(scores)
#print(scores.mean())
accuracies["RFC"]=scores.max()
prediction=cross_val_predict(clf,reduced_features,labels_resnet)
cm=confusion_matrix(labels_resnet,prediction)
print(cm)

##########################################################################################################################################################################


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 7).fit(xtrain, ytrain)
 
# accuracy on X_test
accuracy3= knn.score(xtest, ytest)
accuracy_train3=knn.score(xtrain,ytrain)
print('Training accuracy', accuracy_train3)

print('Test Accuracy',accuracy3)

 
# creating a confusion matrix
knn_predictions = knn.predict(xtest)

print('Prediction is: ',categories[knn_predictions[0]])

#myimage=xtest[0].reshape(63,60)
#plt.imshow(myimage,cmap='gray')
#plt.show()
cm = confusion_matrix(ytest, knn_predictions)
print(cm)

import seaborn as sns
group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in cm.flatten()/np.sum(cm)]
labels = [f"{v1}\n{v2}\n" for v1, v2 in zip(group_counts,group_percentages)]
n=len(categories)
labels = np.asarray(labels).reshape(n,n)
ax = sns.heatmap(cm, annot=labels, fmt='', cmap='Blues')
ax.set_title(f'Seaborn Confusion Matrix with labels for KNeighborsClassifier with Accuracy of {accuracy3*100:.02f}%\n');
ax.set_xlabel('\nPredicted Weather Category')
ax.set_ylabel('Actual Weather Category ');
## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(categories)
ax.yaxis.set_ticklabels(categories)
## Display the visualization of the Confusion Matrix.
plt.show()

from sklearn.model_selection import cross_val_score,cross_val_predict
clf = KNeighborsClassifier(n_neighbors = 7)
scores = cross_val_score(clf, reduced_features, labels_resnet, cv=10)

print(scores)
#print(scores.mean())
accuracies["KNN"]=scores.max()
prediction=cross_val_predict(clf,reduced_features,labels_resnet)
cm=confusion_matrix(labels_resnet,prediction)
print(cm)


##########################################################################################################################################################################


from sklearn.tree import DecisionTreeClassifier
dtree_model = DecisionTreeClassifier(max_depth = k).fit(xtrain, ytrain)
dtree_predictions = dtree_model.predict(xtest)
 
accuracy4=dtree_model.score(xtest,ytest)
accuracy_train4=dtree_model.score(xtrain,ytrain)
print('Training accuracy', accuracy_train4)

print('Test Accuracy',accuracy4)

print('Prediction is: ',categories[dtree_predictions[0]])

# creating a confusion matrix
cm = confusion_matrix(ytest, dtree_predictions)
print(cm)
import seaborn as sns
group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in cm.flatten()/np.sum(cm)]
labels = [f"{v1}\n{v2}\n" for v1, v2 in zip(group_counts,group_percentages)]
n=len(categories)
labels = np.asarray(labels).reshape(n,n)
ax = sns.heatmap(cm, annot=labels, fmt='', cmap='Blues')
ax.set_title(f'Seaborn Confusion Matrix with labels for DecisionTreeClassifier with Accuracy of {accuracy4*100:.02f}%\n');
ax.set_xlabel('\nPredicted Weather Category')
ax.set_ylabel('Actual Weather Category ');
## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(categories)
ax.yaxis.set_ticklabels(categories)
## Display the visualization of the Confusion Matrix.
plt.show()

from sklearn.model_selection import cross_val_score,cross_val_predict
clf = DecisionTreeClassifier(max_depth = 10)
scores = cross_val_score(clf, reduced_features, labels_resnet, cv=10)

print(scores)
accuracies["DTC"]=scores.max()
prediction=cross_val_predict(clf,reduced_features,labels_resnet)
cm=confusion_matrix(labels_resnet,prediction)
print(cm)



##########################################################################################################################################################################



from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB().fit(xtrain, ytrain)
gnb_predictions = gnb.predict(xtest)
 
# accuracy on X_test
accuracy5 = gnb.score(xtest, ytest)
accuracy_train5=gnb.score(xtrain,ytrain)
print('Training accuracy', accuracy_train5)

print('Test Accuracy',accuracy5)


print('Prediction is: ',categories[gnb_predictions[0]])

#myimage=xtest[0].reshape(63,60)
#plt.imshow(myimage)
#plt.show()
 
# creating a confusion matrix
cm = confusion_matrix(ytest, gnb_predictions)
print(cm)
import seaborn as sns
group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in cm.flatten()/np.sum(cm)]
labels = [f"{v1}\n{v2}\n" for v1, v2 in zip(group_counts,group_percentages)]
n=len(categories)
labels = np.asarray(labels).reshape(n,n)
ax = sns.heatmap(cm, annot=labels, fmt='', cmap='Blues')
ax.set_title(f'Seaborn Confusion Matrix with labels for GaussianNB with Accuracy of {accuracy5*100:.02f}%\n');
ax.set_xlabel('\nPredicted Weather Category')
ax.set_ylabel('Actual Weather Category ');
## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(categories)
ax.yaxis.set_ticklabels(categories)
## Display the visualization of the Confusion Matrix.
plt.show()

from sklearn.model_selection import cross_val_score,cross_val_predict
clf = GaussianNB()
scores = cross_val_score(clf, reduced_features, labels_resnet, cv=10)

print(scores)
accuracies["GNB"]=scores.max()
prediction=cross_val_predict(clf,reduced_features,labels_resnet)
cm=confusion_matrix(labels_resnet,prediction)
print(cm)




##########################################################################################################################################################################
'''
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
rfc=RandomForestClassifier(n_estimators=500,max_leaf_nodes=16,n_jobs=-1)
svcc=SVC(C=1,kernel='linear',gamma='auto')
#gnb=GaussianNB()
knn = KNeighborsClassifier(n_neighbors = 7)
lrc=LogisticRegression(max_iter=100000)
dtc=DecisionTreeClassifier(max_depth = 7)
#('gn',gnb)
#,('dt',dtc)
vtc=VotingClassifier(estimators=[('rf',rfc),('sv',svcc),('kn',knn),('lr',lrc)],voting='hard')
vtc.fit(xtrain,ytrain)
accuracy6=vtc.score(xtest,ytest)
accuracy_train6=vtc.score(xtrain,ytrain)
pred=vtc.predict(xtest)
cm=confusion_matrix(ytest,pred)
print(cm)
import seaborn as sns
group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in cm.flatten()/np.sum(cm)]
labels = [f"{v1}\n{v2}\n" for v1, v2 in zip(group_counts,group_percentages)]
n=len(categories)
labels = np.asarray(labels).reshape(n,n)
ax = sns.heatmap(cm, annot=labels, fmt='', cmap='Blues')
ax.set_title(f'Seaborn Confusion Matrix with labels for VotingClassifier with Accuracy of {accuracy6*100:.02f}%\n');
ax.set_xlabel('\nPredicted Weather Category')
ax.set_ylabel('Actual Weather Category ');
## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(categories)
ax.yaxis.set_ticklabels(categories)
## Display the visualization of the Confusion Matrix.
plt.show()

from sklearn.model_selection import cross_val_score,cross_val_predict
clf = VotingClassifier(estimators=[('rf',rfc),('sv',svcc),('kn',knn),('lr',lrc)],voting='hard')
scores = cross_val_score(clf, reduced_features, labels_resnet, cv=10)

print(scores)
accuracies["VTC"]=[scores.max(),accuracy_train6]
prediction=cross_val_predict(clf,reduced_features,labels_resnet)
cm=confusion_matrix(labels_resnet,prediction)
print(cm)

'''
print(accuracies)

models = list(accuracies.keys())
acc = list(accuracies.values())
new=pd.DataFrame()
new['Model']=pd.DataFrame(models)
new['Accuracy']=pd.DataFrame(acc)
#new=pd.DataFrame(accuracies,columns=['Model','Accuracy'])
plt.figure(figsize = (10, 5))
fig=sns.barplot(x="Model",y="Accuracy",data=new)
for bar in fig.patches:
   
  # Using Matplotlib's annotate function and
  # passing the coordinates where the annotation shall be done
  # x-coordinate: bar.get_x() + bar.get_width() / 2
  # y-coordinate: bar.get_height()
  # free space to be left to make graph pleasing: (0, 8)
  # ha and va stand for the horizontal and vertical alignment
    fig.annotate(format(bar.get_height(), '.2f'),
                   (bar.get_x() + bar.get_width() / 2,
                    bar.get_height()), ha='center', va='center',
                   size=15, xytext=(0, 8),
                   textcoords='offset points')
 
# creating the bar plot
#plt.bar(models, acc, color ='blue',
        #width = 0.4)
#plt.anotate
 
#plt.xlabel("Classifiers")
#plt.ylabel("Accuracy")
plt.title("Accuracies For Each Classifier")
plt.show()
