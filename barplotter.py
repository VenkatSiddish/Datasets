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
import seaborn as sns

accuracies={'SVM': 0.984, 'RFC': 0.944, 'KNN': 0.944, 'DTC': 0.88, 'GNB': 0.832}

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
