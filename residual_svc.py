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

from sklearn.preprocessing import normalize
import timm
dir='/home/ameeth/WC/1125/Multi-class Weather Dataset'

categories=['Cloudy','Rain','Shine','Sunrise']
target_names=['Cloudy','Rain','Shine','Sunrise']
#categories=['Cloud','Rain','Sandstorm','Sunrise','Foggy']

data=[]

# Load the pretrained model
model = models.densenet121(pretrained=True)
#print(model)
model.fc=nn.Identity()

# Use the model object to select the desired layer
layer = model._modules.get('avgpool')
# Set model to evaluation mode
model.eval()
scaler = transforms.Resize(size=(224, 224))
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
to_tensor = transforms.ToTensor()

def get_vector(image_name):
    # 1. Load the image with Pillow library
    img = Image.open(image_name) 
    img = img.convert("RGB")  
    #print(img.getbands())
    #img.show()
    # 2. Create a PyTorch Variable with the transformed image
    t_img = Variable(normalize(to_tensor(scaler(img))).unsqueeze(0))    
    # 3. Create a vector of zeros that will hold our feature vector
    #    The 'avgpool' layer has an output size of 512
    my_embedding = torch.zeros(1,2048)    
    # 4. Define a function that will copy the output of a layer
    
    #def copy_data(m, i, o):
        #my_embedding.copy_(o.data)    
    # 5. Attach that function to our selected layer
    #h = layer.register_forward_hook(copy_data)    
    # 6. Run the model on our transformed image
    #model(t_img)    
    # 7. Detach our copy function from the layer
    #h.remove() 
    
    # 8. Return the feature vector
    my_embedding=model(t_img)
    return my_embedding

count=0
for category in categories:
	path= os.path.join(dir, category)
	label=categories.index(category)
	
	for image in os.listdir(path):
			image_path=os.path.join(path,image)
			image_vector=get_vector(image_path)
			image_list_vector=image_vector.tolist()
			image_numpy_vector=np.array(image_list_vector,dtype=object)
			#print(image_numpy_vector.shape)
			row=image_numpy_vector.shape[0];
			col=image_numpy_vector.shape[1];
			#print(row,col)
			print(count,end=' ')
			count=count+1
			image_new=np.reshape(image_numpy_vector,(row*col));
			image_new1=image_new.astype('float')
			data.append([image_new1,label])
		
		
print(len(data))		

pick_in=open('dense121_1125.pickle','wb')
pickle.dump(data,pick_in)
pick_in.close()
exit()
##################################################################################################################################################

pick_in=open('res50_1125.pickle','rb')
data1=pickle.load(pick_in)
pick_in.close()

#print(len(data1))
#print(len(data1[0]))
#print(len(data1[0][0]),len(data1[0][0][0]),len(data1[0][0][0][0]),len(data1[0][0][0][0][0]))
random.shuffle(data1)
features=[]
labels=[]
import pandas as pd
for feature,label in data1:
	features.append(feature)
	labels.append(label)


print(len(features[1]))
df=pd.DataFrame(features)
nor=normalize(df)
norm=pd.DataFrame(nor)
norm = norm.loc[:, (norm != 0).any(axis=0)]
print(norm.shape)
print(len(labels))
##################################################################################################################################################



#from sklearn.decomposition import PCA
#pca=PCA(n_components=0.95)
#px=pca.fit_transform(features)
#x_train_new,x_test_new, y_train,y_test=train_test_split(px,labels,test_size=0.25)


#########################################################################################################################################

#xtrain,xtest, ytrain,ytest=train_test_split(px,labels,test_size=0.25,random_state=42)


#print(len(xtrain[0]),len(xtest[0]),len(ytrain),len(ytest))

#print(len(xtrain), len(xtrain[0]), len(xtrain[0][0]),len(xtrain[0][0][0]),len(xtrain[0][0][0][0]))
#print(len(xtest), len(xtest[0]), len(xtest[0][0]),len(xtest[0][0][0]),len(xtest[0][0][0][0]))
#print(len(ytrain))
#x_train=np.array(xtrain, dtype=object)
#y_train=np.array(ytrain, dtype=object)
#print(x_train.shape)
#print(y_train.shape)
#x_test=np.array(xtest, dtype=object)
#y_test=np.array(ytest, dtype=object)
#print(x_test.shape)
#print(y_test.shape)

#od_train=x_train.shape[0]
#td_train=x_train.shape[2]
#od_test=x_test.shape[0]
#td_test=x_test.shape[2]

#x_train_new=np.reshape(x_train,(od_train,td_train))
#x_test_new=np.reshape(x_test,(od_test,td_test))
#x_train_new=x_train_new.astype('int')
#x_test_new=x_test_new.astype('int')
#y_train=y_train.astype('int')
#y_test=y_test.astype('int')
#print(type(x_train_new))
#print(x_train_new.shape, x_test_new.shape)


#print(x_train_new.shape, x_test_new.shape)

#########################################################################################################################################

from sklearn.feature_selection import SelectPercentile as SP
selector = SP(percentile=70) # select features with top 50% MI scores

selector.fit(norm,labels)
X = selector.transform(norm)
print(X.shape)
'''
x_train_new,x_test_new,y_train,y_test = train_test_split(
    X,labels
    ,random_state=42
    
)
'''
#model_4 = RandomForestClassifier().fit(X_train_4,y_train)
#score_4 = model_4.score(X_test_4,y_test)
#print(f"score_4:{score_4}")

from sklearn.decomposition import PCA
pca=PCA(n_components=0.95)
px=pca.fit_transform(X)
print(px.shape)
xtrain,xtest, ytrain,ytest=train_test_split(px,labels,test_size=0.25,random_state=42)
#x_train_new,x_test_new, y_train,y_test=train_test_split(px,labels,test_size=0.25)
x_train_new,x_test_new, y_train,y_test=xtrain,xtest, ytrain,ytest
#########################################################################################################################################

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
#scores = cross_val_score(clf, X_4, labels_resnet, cv=10)
#scores = cross_val_score(clf, reduced_features, labels_resnet, cv=10)
scores = cross_val_score(clf, px, labels, cv=10)
print(scores)
#accuracies["SVM"]=max(scores.max(),accuracy1)
#prediction=cross_val_predict(clf,reduced_features,labels_resnet)
prediction=cross_val_predict(clf,px,labels)
#prediction=cross_val_predict(clf,X_4,labels_resnet)
cm=confusion_matrix(labels,prediction)
print(cm)

from sklearn.metrics import classification_report

print(classification_report(ytest, svc_prediction, target_names=target_names))

print(time.time()-start)


#########################################################################################################################################



from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 7).fit(x_train_new, y_train)
 
# accuracy on X_test
accuracy = knn.score(x_test_new, y_test)
print(accuracy)
 
# creating a confusion matrix
knn_predictions = knn.predict(x_test_new)

print('Prediction is: ',categories[knn_predictions[0]])

#myimage=xtest[0].reshape(63,60)
#plt.imshow(myimage,cmap='gray')
#plt.show()
cm = confusion_matrix(y_test, knn_predictions)
print(cm)
import seaborn as sns
group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in cm.flatten()/np.sum(cm)]
labels = [f"{v1}\n{v2}\n" for v1, v2 in zip(group_counts,group_percentages)]
labels = np.asarray(labels).reshape(4,4)
ax = sns.heatmap(cm, annot=labels, fmt='', cmap='Blues')
ax.set_title(f'Seaborn Confusion Matrix with labels for KNeighborsClassifier with Accuracy of {accuracy}\n\n');
ax.set_xlabel('\nPredicted Weather Category')
ax.set_ylabel('Actual Weather Category ');
## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(categories)
ax.yaxis.set_ticklabels(categories)
## Display the visualization of the Confusion Matrix.
plt.show()

#########################################################################################################################################




from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier()
model.fit(x_train_new,y_train)
forest_predictions= model.predict(x_test_new)
accuracy=model.score(x_test_new,y_test)
print(accuracy)


print('Prediction is: ',categories[forest_predictions[0]])

#myimage=xtest[0].reshape(63,60)
#plt.imshow(myimage)
#plt.show()

# creating a confusion matrix
cm = confusion_matrix(y_test, forest_predictions)
print(cm)
import seaborn as sns
group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in cm.flatten()/np.sum(cm)]
labels = [f"{v1}\n{v2}\n" for v1, v2 in zip(group_counts,group_percentages)]
labels = np.asarray(labels).reshape(4,4)
ax = sns.heatmap(cm, annot=labels, fmt='', cmap='Blues')
ax.set_title(f'Seaborn Confusion Matrix with labels for RFClassifier with Accuracy of {accuracy}\n\n');
ax.set_xlabel('\nPredicted Weather Category')
ax.set_ylabel('Actual Weather Category ');
## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(categories)
ax.yaxis.set_ticklabels(categories)
## Display the visualization of the Confusion Matrix.
plt.show()


#########################################################################################################################################


from sklearn.tree import DecisionTreeClassifier
dtree_model = DecisionTreeClassifier(max_depth = 10).fit(x_train_new, y_train)
dtree_predictions = dtree_model.predict(x_test_new)
 
acurracy=dtree_model.score(x_test_new,y_test)
print('Accuracy',acurracy)

print('Prediction is: ',categories[dtree_predictions[0]])

#myimage=xtest[0].reshape(63,60)
#plt.imshow(myimage,cmap='gray')
#plt.show()
# creating a confusion matrix
cm = confusion_matrix(y_test, dtree_predictions)
print(cm)
import seaborn as sns
group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in cm.flatten()/np.sum(cm)]
labels = [f"{v1}\n{v2}\n" for v1, v2 in zip(group_counts,group_percentages)]
labels = np.asarray(labels).reshape(4,4)
ax = sns.heatmap(cm, annot=labels, fmt='', cmap='Blues')
ax.set_title(f'Seaborn Confusion Matrix with labels for KNeighborsClassifier with Accuracy of {accuracy}\n\n');
ax.set_xlabel('\nPredicted Weather Category')
ax.set_ylabel('Actual Weather Category ');
## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(categories)
ax.yaxis.set_ticklabels(categories)
## Display the visualization of the Confusion Matrix.
plt.show()

#########################################################################################################################################


from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB().fit(x_train_new, y_train)
gnb_predictions = gnb.predict(x_test_new)
 
# accuracy on X_test
accuracy = gnb.score(x_test_new, y_test)
print(accuracy)


print('Prediction is: ',categories[gnb_predictions[0]])

#myimage=xtest[0].reshape(63,60)
#plt.imshow(myimage)
#plt.show()
 
# creating a confusion matrix
cm = confusion_matrix(y_test, gnb_predictions)
print(cm)
import seaborn as sns
group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in cm.flatten()/np.sum(cm)]
labels = [f"{v1}\n{v2}\n" for v1, v2 in zip(group_counts,group_percentages)]
labels = np.asarray(labels).reshape(4,4)
ax = sns.heatmap(cm, annot=labels, fmt='', cmap='Blues')
ax.set_title(f'Seaborn Confusion Matrix with labels for KNeighborsClassifier with Accuracy of {accuracy}\n\n');
ax.set_xlabel('\nPredicted Weather Category')
ax.set_ylabel('Actual Weather Category ');
## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(categories)
ax.yaxis.set_ticklabels(categories)
## Display the visualization of the Confusion Matrix.
plt.show()

