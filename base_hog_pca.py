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
import time

#Directory and Categories for classification
dir= '/home/ameeth/WC/'


categories=['Cloudy','Rain','Shine','Sunrise']
target_names = ['Cloudy','Rain','Shine','Sunrise']
#categories=['HAZE','RAINY','SUNNY','SNOWY']
#target_names=['HAZE','RAINY','SUNNY','SNOWY']
#Opening feature pickle files

pick_in=open('resNet101_1125.pickle','rb')
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

#normed=np.concatenate((norm1,norm2),axis=1)

from sklearn.decomposition import PCA
pca=PCA(n_components=0.95)
#px1=pca1.fit_transform(X_4_1)
#px2=pca.fit_transform(X_4_2)
px1=pca.fit_transform(norm1)
px2=pca.fit_transform(norm2)

print(px1.shape,px2.shape)


#Feature Selection Using Mututal Information

from sklearn.feature_selection import SelectPercentile as SP
selector1 = SP(percentile=90) # select features with top 50% MI scores

selector1.fit(px1,labels_new)
X_4_1 = selector1.transform(px1)
#X_4_1 = selector1.transform(px1)
print(X_4_1.shape, type(X_4_1))

selector2 = SP(percentile=90) # select features with top 50% MI scores
selector2.fit(px2,labels_new)
X_4_2 = selector2.transform(px2)
print(X_4_2.shape, type(X_4_2))

'''
from sklearn.feature_selection import SelectPercentile as SP
selector1 = SP(percentile=50) # select features with top 50% MI scores

selector1.fit(norm1,labels_new)
X_4_1 = selector1.transform(norm1)
#X_4_1 = selector1.transform(px1)
print(X_4_1.shape, type(X_4_1))

selector2 = SP(percentile=50) # select features with top 50% MI scores
selector2.fit(norm2,labels_new)
X_4_2 = selector2.transform(norm2)
print(X_4_2.shape, type(X_4_2))
#normed=np.concatenate((X_4_1,X_4_2),axis=1)

#Feature Reduction using PCA

from sklearn.decomposition import PCA
pca=PCA(n_components=0.95)
px1=pca.fit_transform(X_4_1)
px2=pca.fit_transform(X_4_2)
#px1=pca.fit_transform(norm1)
#px2=pca.fit_transform(norm2)

print(px1.shape,px2.shape)
'''

normed=np.concatenate((px1,px2),axis=1)
#normed=np.concatenate((X_4_1,X_4_2),axis=1)


df_new=pd.DataFrame(normed)
df_new['labels']=pd.DataFrame(labels_new)
print(df_new.head())
shuffled = df_new.sample(frac=1,random_state=42).reset_index()
shuffled.drop(shuffled.columns[0],axis=1,inplace=True)
print(shuffled.head())
labels=shuffled.iloc[:,-1]
print(labels.head())
norm=shuffled.drop(shuffled.columns[-1],axis=1)
print(norm.head())





#xtrain,xtest, ytrain,ytest=train_test_split(norm,labels,test_size=0.2,random_state=42)
xtrain,xtest, ytrain,ytest=train_test_split(norm,labels,test_size=0.25,random_state=42)
accuracies={}
mean_accuracies={}
##########################################################################################################################################################################
#SVC 
'''
import time
start=time.time()
from sklearn.svm import SVC
model=SVC(C=1,kernel='linear',gamma='auto')
model.fit(xtrain,ytrain)
svc_prediction=model.predict(xtest)
accuracy1=model.score(xtest,ytest)
accuracy_train1=model.score(xtrain,ytrain)

print('Training accuracy', accuracy_train1)

print('Test Accuracy',accuracy1)
exit()
print('Prediction is: ',categories[svc_prediction[0]])
cm=confusion_matrix(ytest,svc_prediction)
print(cm)

import seaborn as sns
group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in cm.flatten()/np.sum(cm)]
lab = [f"{v1}\n{v2}\n" for v1, v2 in zip(group_counts,group_percentages)]
n=len(categories)
lab = np.asarray(lab).reshape(n,n)
ax = sns.heatmap(cm, annot=lab, fmt='', cmap='Blues')
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

scores = cross_val_score(clf, norm, labels, cv=5)
#scores = cross_val_score(clf, norm, labels, cv=10)

print(scores.mean())

accuracies["SVM"]=max(scores.max(),accuracy1)
mean_accuracies["SVM"]=scores.mean()
prediction=cross_val_predict(clf,norm,labels)
#prediction=cross_val_predict(clf,px,labels_resnet)
#prediction=cross_val_predict(clf,X_4,labels_resnet)
cm=confusion_matrix(labels,prediction)
print(cm)

from sklearn.metrics import classification_report

print(classification_report(ytest, svc_prediction, target_names=target_names))

print(time.time()-start)
'''
##########################################################################################################################################################################
#Random Forest Classifier


start=time.time()
k=6
from sklearn.ensemble import RandomForestClassifier
'''
from sklearn.model_selection import GridSearchCV
# defining parameter range
param_grid = {  'bootstrap': [True], 'max_depth': [5,6,7,8], 'max_features': ['auto', 'log2'], 'n_estimators': [100,200,300,400,500]}
model = GridSearchCV(RandomForestClassifier(), param_grid, refit = True, verbose = 3,scoring='accuracy')
# fitting the model for grid search
model.fit(xtrain, ytrain)
print(model.best_params_)
print(model.best_score_)
exit()
'''
rfc_model=RandomForestClassifier(bootstrap= True, max_depth=8, max_features= 'auto', n_estimators= 300)
rfc_model.fit(xtrain,ytrain)
forest_predictions= rfc_model.predict(xtest)
accuracy2=rfc_model.score(xtest,ytest)
accuracy_train2=rfc_model.score(xtrain,ytrain)
print('Training accuracy', accuracy_train2)

print('Test Accuracy',accuracy2)

print('Prediction is: ',categories[forest_predictions[0]])



# creating a confusion matrix
cm = confusion_matrix(ytest, forest_predictions)
print(cm)

'''
import seaborn as sns
group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in cm.flatten()/np.sum(cm)]
lab = [f"{v1}\n{v2}\n" for v1, v2 in zip(group_counts,group_percentages)]
n=len(categories)
lab = np.asarray(lab).reshape(n,n)
ax = sns.heatmap(cm, annot=lab, fmt='', cmap='Blues')
ax.set_title(f'Seaborn Confusion Matrix with labels for Random Forest Classifier with Accuracy of {accuracy2*100:.02f}%\n');
ax.set_xlabel('\nPredicted Weather Category')
ax.set_ylabel('Actual Weather Category ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(categories)
ax.yaxis.set_ticklabels(categories)

## Display the visualization of the Confusion Matrix.
plt.show()
'''
'''
from sklearn.model_selection import cross_val_score,cross_val_predict
clf = RandomForestClassifier()

scores = cross_val_score(clf, norm, labels, cv=5)
#scores = cross_val_score(clf, norm, labels, cv=10)
print(scores.mean())
accuracies["RFC"]=max(scores.max(),accuracy2)
mean_accuracies["RFC"]=scores.mean()
prediction=cross_val_predict(clf,norm,labels)

cm=confusion_matrix(labels,prediction)
print(cm)

from sklearn.metrics import classification_report

print(classification_report(ytest, forest_predictions, target_names=target_names))
print(time.time()-start)
'''
##########################################################################################################################################################################
#K Nearest Neighbors

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 10).fit(xtrain, ytrain)
 
# accuracy on X_test
accuracy3= knn.score(xtest, ytest)
accuracy_train3=knn.score(xtrain,ytrain)
print('Training accuracy', accuracy_train3)

print('Test Accuracy',accuracy3)

 
# creating a confusion matrix
knn_predictions = knn.predict(xtest)

print('Prediction is: ',categories[knn_predictions[0]])

cm = confusion_matrix(ytest, knn_predictions)
print(cm)
exit()
'''
import seaborn as sns
group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in cm.flatten()/np.sum(cm)]
lab = [f"{v1}\n{v2}\n" for v1, v2 in zip(group_counts,group_percentages)]
n=len(categories)
lab = np.asarray(lab).reshape(n,n)
ax = sns.heatmap(cm, annot=lab, fmt='', cmap='Blues')
ax.set_title(f'Seaborn Confusion Matrix with labels for KNeighborsClassifier with Accuracy of {accuracy3*100:.02f}%\n');
ax.set_xlabel('\nPredicted Weather Category')
ax.set_ylabel('Actual Weather Category ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(categories)
ax.yaxis.set_ticklabels(categories)

## Display the visualization of the Confusion Matrix.
plt.show()
'''
from sklearn.model_selection import cross_val_score,cross_val_predict
clf = KNeighborsClassifier(n_neighbors = 7)
scores = cross_val_score(clf, norm, labels, cv=5)
#scores = cross_val_score(clf, norm, labels, cv=10)
print(scores.mean())

accuracies["KNN"]=max(scores.max(),accuracy3)
mean_accuracies["KNN"]=scores.mean()
prediction=cross_val_predict(clf,norm,labels)

cm=confusion_matrix(labels,prediction)
print(cm)

from sklearn.metrics import classification_report

print(classification_report(ytest, knn_predictions, target_names=target_names))
##########################################################################################################################################################################
#Decision Tree

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
'''
import seaborn as sns
group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in cm.flatten()/np.sum(cm)]
lab = [f"{v1}\n{v2}\n" for v1, v2 in zip(group_counts,group_percentages)]
n=len(categories)
lab = np.asarray(lab).reshape(n,n)
ax = sns.heatmap(cm, annot=lab, fmt='', cmap='Blues')
ax.set_title(f'Seaborn Confusion Matrix with labels for DecisionTreeClassifier with Accuracy of {accuracy4*100:.02f}%\n');
ax.set_xlabel('\nPredicted Weather Category')
ax.set_ylabel('Actual Weather Category ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(categories)
ax.yaxis.set_ticklabels(categories)

## Display the visualization of the Confusion Matrix.
plt.show()
'''
from sklearn.model_selection import cross_val_score,cross_val_predict
clf = DecisionTreeClassifier(max_depth = 10)

scores = cross_val_score(clf, norm, labels, cv=5)
#scores = cross_val_score(clf, norm, labels, cv=10)


print(scores.mean())
accuracies["DTC"]=max(scores.max(),accuracy4)
mean_accuracies["DTC"]=scores.mean()
prediction=cross_val_predict(clf,norm,labels)
cm=confusion_matrix(labels,prediction)
print(cm)

from sklearn.metrics import classification_report

print(classification_report(ytest, dtree_predictions, target_names=target_names))

##########################################################################################################################################################################
#Naive Bayes


from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB().fit(xtrain, ytrain)
gnb_predictions = gnb.predict(xtest)
 
# accuracy on X_test
accuracy5 = gnb.score(xtest, ytest)
accuracy_train5=gnb.score(xtrain,ytrain)
print('Training accuracy', accuracy_train5)

print('Test Accuracy',accuracy5)


print('Prediction is: ',categories[gnb_predictions[0]])
 
# creating a confusion matrix
cm = confusion_matrix(ytest, gnb_predictions)
print(cm)
'''
import seaborn as sns
group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in cm.flatten()/np.sum(cm)]
lab = [f"{v1}\n{v2}\n" for v1, v2 in zip(group_counts,group_percentages)]
n=len(categories)
lab = np.asarray(lab).reshape(n,n)
ax = sns.heatmap(cm, annot=lab, fmt='', cmap='Blues')
ax.set_title(f'Seaborn Confusion Matrix with labels for GaussianNB with Accuracy of {accuracy5*100:.02f}%\n');
ax.set_xlabel('\nPredicted Weather Category')
ax.set_ylabel('Actual Weather Category ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(categories)
ax.yaxis.set_ticklabels(categories)

## Display the visualization of the Confusion Matrix.
plt.show()
'''
from sklearn.model_selection import cross_val_score,cross_val_predict
clf = GaussianNB()
scores = cross_val_score(clf, norm, labels, cv=5)
#scores = cross_val_score(clf, norm, labels, cv=10)
print(scores.mean())
accuracies["GNB"]=max(scores.max(),accuracy5)
mean_accuracies["GNB"]=scores.mean()
prediction=cross_val_predict(clf,norm,labels)

cm=confusion_matrix(labels,prediction)
print(cm)


from sklearn.metrics import classification_report

print(classification_report(ytest, gnb_predictions, target_names=target_names))

##########################################################################################################################################################################
#Accuracies for each classifier plot
import seaborn as sns
print(accuracies)
print(mean_accuracies)

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
    fig.annotate(format(bar.get_height(), '.5f'),
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



