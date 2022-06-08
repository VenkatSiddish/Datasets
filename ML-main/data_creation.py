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
categories=['Cloud','Rain','Sandstorm','Sunrise','Foggy']

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

pick_in=open('5sandstorm.pickle','wb')
pickle.dump(data,pick_in)
pick_in.close()

##########################################################################################################################################################################
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

#categories=['Cloud','Rain','Shine','Sunrise','Snow','Sandstorm','Foggy','Fog_Smog','Lightning']
categories=['Cloud','Rain','Sandstorm','Sunrise','Foggy']
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
pick_in=open('5sandstorm_hog.pickle','wb')
pickle.dump(data,pick_in)
pick_in.close()
