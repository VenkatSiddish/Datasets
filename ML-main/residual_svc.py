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

#categories=['Cloud','Rain','Shine','Sunrise']
categories=['Cloud','Rain','Shine','Sunrise','Foggy']

data=[]

# Load the pretrained model
model = models.resnet18(pretrained=True)
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
			image_new=np.reshape(image_numpy_vector,(row*col));
			image_new1=image_new.astype('float')
			data.append([image_new1,label])
		
		
print(len(data))		

pick_in=open('data5_deep.pickle','wb')
pickle.dump(data,pick_in)
pick_in.close()

##################################################################################################################################################

pick_in=open('data5_deep.pickle','rb')
data1=pickle.load(pick_in)
pick_in.close()

#print(len(data1))
#print(len(data1[0]))
#print(len(data1[0][0]),len(data1[0][0][0]),len(data1[0][0][0][0]),len(data1[0][0][0][0][0]))
random.shuffle(data1)
features=[]
labels=[]

for feature,label in data1:
	features.append(feature)
	labels.append(label)


##################################################################################################################################################

'''
xtrain=np.array(features, dtype=object)
ytrain=np.array(labels, dtype=object)

od_train=xtrain.shape[0]
td_train=xtrain.shape[2]


x_trainnew=np.reshape(xtrain,(od_train,td_train))
x_trainnew=x_trainnew.astype('int')
ytrain=ytrain.astype('int')


from sklearn.decomposition import PCA
pca=PCA(n_components=0.95)
px=pca.fit_transform(x_trainnew)
x_train_new,x_test_new, y_train,y_test=train_test_split(px,ytrain,test_size=0.25)
'''
#########################################################################################################################################

x_train_new,x_test_new, y_train,y_test=train_test_split(features,labels,test_size=0.25)
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

from sklearn.svm import SVC
model=SVC(C=1,kernel='linear',gamma='auto')
model.fit(x_train_new,y_train)

prediction=model.predict(x_test_new)

acurracy=model.score(x_test_new,y_test)
print('Accuracy',acurracy)

print('Prediction is: ',categories[prediction[0]])
cm=confusion_matrix(y_test,prediction)
print(cm)

#########################################################################################################################################


'''
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
'''
#########################################################################################################################################

'''


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
'''

#########################################################################################################################################

'''

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
'''
#########################################################################################################################################


'''

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
'''
