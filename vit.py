import os
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import random
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable

dir= '/home/ameeth/WC'
categories=['Cloud','Rain','Sunrise','Sandstorm','Foggy']

from transformers import ViTFeatureExtractor

#model_name_or_path = 'google/vit-base-patch16-224-in21k'

model_name_or_path = 'https://storage.googleapis.com/vit_models/imagenet21k/R50%2BViT-L_32.npz'

#model_name_or_path = '/home/ameeth/WC/R50+ViT-L_32.npz'
#model=np.load(model_name_or_path)

feature_extractor = ViTFeatureExtractor.from_pretrained(model_name_or_path,encoding='windows-1252')
#feature_extractor = ViTFeatureExtractor.from_pretrained(model)

print(feature_extractor)


data=[]
for category in categories:
	path= os.path.join(dir, category)
	label=categories.index(category)
	
	for image in os.listdir(path):

		image_path=os.path.join(path,image)
		img=Image.open(image_path)
		img=img.convert('RGB')
		feat=feature_extractor(img)
		print(feat)
		break
		feats=np.array(list(feat.values()))
		print(feats.shape)
		#feature=np.reshape(feats,(224*3*224))
		break
		data.append([feature,label])
		
	break
'''
pick_in=open('vit_data5_new.pickle','wb')
pickle.dump(data,pick_in)
pick_in.close()


pick_in=open('vit_data5_new.pickle','rb')
data1=pickle.load(pick_in)
pick_in.close()

random.shuffle(data1)
features=[]

labels=[]

for feature,label in data1:
	features.append(feature)
	labels.append(label)
	
from sklearn.decomposition import PCA
pca=PCA(n_components=0.95)
px=pca.fit_transform(features)
xtrain,xtest, ytrain,ytest=train_test_split(px,labels,test_size=0.25)
#xtrain,xtest, ytrain,ytest=train_test_split(features,labels,test_size=0.25)

from sklearn.svm import SVC
model=SVC(C=1,kernel='linear',gamma='auto')
model.fit(xtrain,ytrain)
prediction=model.predict(xtest)
accuracy=model.score(xtest,ytest)

print('Accuracy',accuracy)
categories=['Cloud','Rain','Sandstorm','Sunrise','Foggy']
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
ax.set_title(f'Seaborn Confusion Matrix with labels for SVM with Accuracy of {accuracy}\n\n');
ax.set_xlabel('\nPredicted Weather Category')
ax.set_ylabel('Actual Weather Category ');
## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(categories)
ax.yaxis.set_ticklabels(categories)
## Display the visualization of the Confusion Matrix.
plt.show()


'''
