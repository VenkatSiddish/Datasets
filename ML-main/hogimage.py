import cv2
from skimage.feature import hog
from skimage.transform import resize
from skimage.io import imread, imshow
import matplotlib.pyplot as plt
from skimage import exposure

image=imread("rain1.jpg")
#cv2.imshow('original',image)
#cv2.waitKey(0)
#print(imshow(image))
#plt.imshow(image)
#plt.show()

resized_image=resize(image,(128,64))
fd,hog_image=hog(resized_image,orientations=9,pixels_per_cell=(8,8),cells_per_block=(2,2),visualize=True,multichannel=True)
print(fd.shape)
print(fd)
#hog_image_rescaled=exposure.rescale_intensity(hog_image,in_range=(0,10))
#plt.imshow(hog_image_rescaled,cmap=plt.cm.gray)
#plt.show()
