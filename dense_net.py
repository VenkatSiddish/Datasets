import tensorflow
from tensorflow.keras.layers import Input,BatchNormalization,ReLU,Conv2D,Dense,MaxPool2D,AvgPool2D,GlobalAvgPool2D,Concatenate
from keras.applications.densenet import preprocess_input, DenseNet121
from PIL import Image

def bn_relu_conv(x,filters,kernel_size):
	x=BatchNormalization()(x)
	x=ReLU()(x)
	x=Conv2D(filters=filters,kernel_size=kernel_size,padding='same')(x)
	
	return x

def dense_block(tensor,k,reps):
	for _ in range(reps):
		x=bn_relu_conv(tensor,filters=4*k,kernel_size=1)
		x=bn_relu_conv(x,filters=k,kernel_size=3)
		tensor=Concatenate()([tensor,x])
	return tensor

def transition_layer(x,theta):
	f=int(tensorflow.keras.backend.int_shape(x)[-1]*theta)
	x=bn_relu_conv(x,filters=f,kernel_size=1)
	x=AvgPool2D(pool_size=2,strides=2,padding='same')
	return x


k=32
theta=0.5
repetitions= 6, 12, 24, 16

input=Input(shape=(224,224,3))
x=Conv2D(2*k,7,strides=2,padding='same')(input)
x=MaxPool2D(3,strides=2,padding='same')(x)

for reps in repetitions:
	d=dense_block(x,k,reps)
	x=transition_layer(d,theta)
x=GlobalAvgPool()(d)
output=Dense(1000,activation='softmax')(x)

from tensorflow.keras import Model
model=Model(input, output)
	
img=Image.open('rain20.jpg')
img=img.convert("RGB")
resized_img=img.resize((224,224))
new_image=preprocess_input(resized_img)
features=model.predict(new_image)

'''
import cv2
import pandas as pd
import numpy as np
import os
from tqdm import tqdm, tqdm_notebook

train_df = pd.read_csv('../input/train/train.csv')
img_size = 256
batch_size = 16

pet_ids = train_df['PetID'].values
n_batches = len(pet_ids) // batch_size + 1

from keras.applications.densenet import preprocess_input, DenseNet121

def resize_to_square(im):
    old_size = im.shape[:2] # old_size is in (height, width) format
    ratio = float(img_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    # new_size should be in (width, height) format
    im = cv2.resize(im, (new_size[1], new_size[0]))
    delta_w = img_size - new_size[1]
    delta_h = img_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,value=color)
    return new_im

def load_image(path, pet_id):
    image = cv2.imread(f'{path}{pet_id}-1.jpg')
    new_image = resize_to_square(image)
    new_image = preprocess_input(new_image)
    return new_image

from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Input, Lambda, AveragePooling1D
import keras.backend as K
inp = Input((256,256,3))
backbone = DenseNet121(input_tensor = inp, include_top = False)
x = backbone.output
x = GlobalAveragePooling2D()(x)
x = Lambda(lambda x: K.expand_dims(x,axis = -1))(x)
x = AveragePooling1D(4)(x)
out = Lambda(lambda x: x[:,:,0])(x)

m = Model(inp,out)

features = {}
for b in tqdm_notebook(range(n_batches)):
    start = b*batch_size
    end = (b+1)*batch_size
    batch_pets = pet_ids[start:end]
    batch_images = np.zeros((len(batch_pets),img_size,img_size,3))
    for i,pet_id in enumerate(batch_pets):
        try:
            batch_images[i] = load_image("../input/train_images/", pet_id)
        except:
            pass
    batch_preds = m.predict(batch_images)
    for i,pet_id in enumerate(batch_pets):
        features[pet_id] = batch_preds[i]


'''
