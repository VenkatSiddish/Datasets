from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pickle
import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import pandas as pd


from skimage.transform import resize
from skimage.feature import hog
from skimage import exposure
from skimage.io import imread, imshow
import pickle
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
from scipy.stats import entropy
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

dir= '/home/ameeth/WC'

categories=['Cloud','Rain','Shine','Sunrise','Snow','Sandstorm','Foggy','Fog_Smog','Lightning']
pick_in=open('data9_hog_res18.pickle','rb')
data1=pickle.load(pick_in)
pick_in.close()

#random.shuffle(data1)
features=[]
labels=[]

for feature,label in data1:
	features.append(feature)
	labels.append(label)

print(len(features[1]))

new=pd.DataFrame(features)
print(type(new))
norm=normalize(new)


def plot2d(component1,component2):
	#fig=plt.Figure(plt.scatter(x=component1,y=component2))
	sns.scatterplot(norm=component1,labels=component2)
	#plt.plot()
	fig.show()
	#return fig

from sklearn.feature_selection import VarianceThreshold

selector=VarianceThreshold(threshold=0.082)
X_train_vrth=selector.fit_transform(features)
#n_features1=X_train_vrth.shape[1]
#print('Features after variance threshold %d with full FV',n_features1)

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

df=pd.DataFrame()

#,perplexity=40,n_iter=400,early_exaggeration=5.0,
per_plex= 36,37,40,42
learn=200.0,250.0,300.0,350.0
'''
for i in per_plex:
	for j in learn:
		tsne=TSNE(random_state=42,n_components=2,perplexity=i,verbose=0,n_iter=1000,early_exaggeration=10.0,learning_rate=j).fit_transform(px)
		df["comp1"]=tsne[:,0]
		df["comp2"]=tsne[:,1]
		df["y"]=labels
		sns.scatterplot(x="comp1",y="comp2",hue=df.y.tolist(),palette=sns.color_palette("hls",4),data=df).set(title=f"t-SNE plot for preplexity {i} and learning rate{j}")	
		plt.legend(categories)
		plt.show()
#print(tsne[:,0])
#print(df.columns)
'''
'''
tsne=TSNE(random_state=42,n_components=2,perplexity=37,verbose=0,n_iter=10000,n_iter_without_progress=500,learning_rate=225.0).fit_transform(px)
df["comp1"]=tsne[:,0]
df["comp2"]=tsne[:,1]
df["y"]=labels
sns.scatterplot(x="comp1",y="comp2",hue=df.y.tolist(),palette=sns.color_palette("hls",9),data=df).set(title="t-SNE plot for preplexity 37 and learning rate 225")	
plt.legend(categories)
plt.show()
'''
'''
#print(tsne[:,1])
#plot2d(tsne[:,0],tsne[:,1])

sns.scatterplot(x="comp1",y="comp2",hue=df.y.tolist(),palette=sns.color_palette("brg",4),data=df).set(title="t-SNE plot")
plt.legend(categories)
plt.show()
'''
#palette=sns.color_palette("hls",3),

#df.y=df.y+1
#print(df.y.tolist())


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
import time
new_std=StandardScaler().fit_transform(new)
df1=pd.DataFrame()
start=time.time()
df1["y"]=labels
x_lda=LDA(n_components=3).fit_transform(new_std,df1.y)
df1["comp1"]=x_lda[:,0]
df1["comp2"]=x_lda[:,1]
sns.scatterplot(x='comp1',y='comp2',hue=df1.y.tolist(),palette=sns.color_palette("rocket_r",9),data=df1).set(title="LDA plot")
plt.legend(categories)
plt.show()	

