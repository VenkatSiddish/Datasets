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

dir= '/home/ameeth/WC'

categories=['Cloud','Rain','Shine','Sunrise','Snow','Sandstorm','Foggy','Fog_Smog','Lightning']
'''
data=[]

# Load the pretrained model
model = models.resnet18(pretrained=True)
# Use the model object to select the desired layer
layer = model._modules.get('avgpool')
# Set model to evaluation mode
model.eval()

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
to_tensor = transforms.ToTensor()

def get_vector(image_name):
    scaler = transforms.Resize(size=(224, 224))
    # 1. Load the image with Pillow library
    img = Image.open(image_name) 
    img = img.convert("RGB")  
    #print(img.getbands())
    #img.show()
    # 2. Create a PyTorch Variable with the transformed image
    t_img = Variable(normalize(to_tensor(scaler(img))).unsqueeze(0))    
    # 3. Create a vector of zeros that will hold our feature vector
    #    The 'avgpool' layer has an output size of 512
    my_embedding = torch.zeros(1,512,1,1)    
    # 4. Define a function that will copy the output of a layer
    def copy_data(m, i, o):
        my_embedding.copy_(o.data)    
    # 5. Attach that function to our selected layer
    h = layer.register_forward_hook(copy_data)    
    # 6. Run the model on our transformed image
    model(t_img)    
    # 7. Detach our copy function from the layer
    h.remove()    
    # 8. Return the feature vector
    return my_embedding
def hog_vector(image_name):
	img=Image.open(image_name)
	img=img.convert("RGB")
	scaler=transforms.Resize(size=(128,64))
	resized_img=scaler(img)
	fd, hog_image = hog(resized_img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True, multichannel=True)
	return fd
	
for category in categories:
	path= os.path.join(dir, category)
	label=categories.index(category)
	
	for image in os.listdir(path):
		features=np.array([],dtype=float)
		image_path=os.path.join(path,image)
		try:
			image_vector=get_vector(image_path)
			image_list_vector=image_vector.tolist()
			image_numpy_vector=np.array(image_list_vector,dtype=object)
			#print(image_numpy_vector.shape)
			row=image_numpy_vector.shape[0];
			col=image_numpy_vector.shape[1];
			#print(row,col)
			image_new=np.reshape(image_numpy_vector,(row*col));
			image_new1=image_new.astype('float')
			#image1=Image.open(image_path)
			#image1=image1.convert("RGB")
			#resized_img=transforms.Resize(image1,(128,64))
			#fd, hog_image = hog(resized_img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True, multichannel=True)
			hog_feature=hog_vector(image_path)
			hog_size=hog_feature.shape[0]
			#print(hog_size)
			features=np.append(features, image_new1,axis=0)
			features=np.append(features,hog_feature,axis=0)
			#resized_feature=features.reshape(row*col*hog_size)
			#print(features.shape)
			data.append([features,label])
		except Exception as e:
			pass




print(len(data))
print(type(data[0]))
pick_in=open('data9_hog_res18.pickle','wb')
pickle.dump(data,pick_in)
pick_in.close()
'''		



pick_in=open('data9_hog_res18.pickle','rb')
data1=pickle.load(pick_in)
pick_in.close()


print(len(data1))
print(len(data1[0]))
print(len(data1[0][0]))



random.shuffle(data1)
features=[]
labels=[]

for feature,label in data1:
	features.append(feature)
	labels.append(label)
print(len(features[0]))
from sklearn.decomposition import PCA
pca=PCA(n_components=0.95)
px=pca.fit_transform(features)
xtrain,xtest, ytrain,ytest=train_test_split(px,labels,test_size=0.25)
#xtrain,xtest, ytrain,ytest=train_test_split(features,labels,test_size=0.25)
print(len(xtrain[0]),len(xtest[0]),len(ytrain),len(ytest))

'''
import umap
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

for clf in (rfc,svcc,knn,lrc,vtc):
	clf.fit(xtrain,ytrain)
	ypred=clf.predict(xtest)
	print(clf.__class__.__name__,clf.score(xtest,ytest))
	cm=confusion_matrix(ytest,ypred)
	print(cm)
	

'''




from sklearn.svm import SVC
model=SVC(C=1,kernel='linear',gamma='auto')


while(True):
	px=pca.fit_transform(features)
	xtrain,xtest, ytrain,ytest=train_test_split(px,labels,test_size=0.25)
	model.fit(xtrain,ytrain)
	prediction=model.predict(xtest)
	acurracy=model.score(xtest,ytest)
	if(acurracy>0.9):
		print('Accuracy',acurracy)
		break

#reducer=umap.uMAP()
#embedding = reducer.fit_transform(xtrain)
#print(embedding.shape)

'''
trans = umap.UMAP(n_neighbors=5, random_state=42).fit(xtrain)
plt.scatter(trans.embedding_[:, 0], trans.embedding_[:, 1], s= 5, c=y_train, cmap='Spectral')
plt.title('Embedding of the training set by UMAP', fontsize=24)
plt.show()
'''
'''	
	
from sklearn.svm import SVC
model=SVC(C=1,kernel='linear',gamma='auto')
model.fit(xtrain,ytrain)
prediction=model.predict(xtest)
acurracy=model.score(xtest,ytest)

print('Accuracy',acurracy)
categories=['Cloud','Rain','Shine','Sunrise']
print('Prediction is: ',categories[prediction[0]])
cm=confusion_matrix(ytest,prediction)
print(cm)

'''

'''

from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier()
model.fit(xtrain,ytrain)
forest_predictions= model.predict(xtest)
accuracy=model.score(xtest,ytest)
print(accuracy)

categories=['Cloud','Rain','Shine','Sunrise']
print('Prediction is: ',categories[forest_predictions[0]])

#myimage=xtest[0].reshape(63,60)
#plt.imshow(myimage)
#plt.show()

# creating a confusion matrix
cm = confusion_matrix(ytest, forest_predictions)
print(cm)


'''

'''

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 7).fit(xtrain, ytrain)
 
# accuracy on X_test
accuracy = knn.score(xtest, ytest)
print(accuracy)
 
# creating a confusion matrix
knn_predictions = knn.predict(xtest)
categories=['Cloud','Rain','Shine','Sunrise']
print('Prediction is: ',categories[knn_predictions[0]])

#myimage=xtest[0].reshape(63,60)
#plt.imshow(myimage,cmap='gray')
#plt.show()
cm = confusion_matrix(ytest, knn_predictions)
print(cm)

'''

'''

from sklearn.tree import DecisionTreeClassifier
dtree_model = DecisionTreeClassifier(max_depth = 10).fit(xtrain, ytrain)
dtree_predictions = dtree_model.predict(xtest)
 
acurracy=dtree_model.score(xtest,ytest)
print('Accuracy',acurracy)
categories=['Cloud','Rain','Shine','Sunrise']
print('Prediction is: ',categories[dtree_predictions[0]])

# creating a confusion matrix
cm = confusion_matrix(ytest, dtree_predictions)
print(cm)


'''


'''

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB().fit(xtrain, ytrain)
gnb_predictions = gnb.predict(xtest)
 
# accuracy on X_test
accuracy = gnb.score(xtest, ytest)
print(accuracy)

categories=['Cloud','Rain','Shine','Sunrise']
print('Prediction is: ',categories[gnb_predictions[0]])

#myimage=xtest[0].reshape(63,60)
#plt.imshow(myimage)
#plt.show()
 
# creating a confusion matrix
cm = confusion_matrix(ytest, gnb_predictions)
print(cm)




'''
