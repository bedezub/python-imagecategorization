#!/usr/bin/env python
# coding: utf-8

# In[79]:


# Intialization Section

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

## Directory of the datasets, in this case the images
DATADIR = "/Users/zubquzaini/Documents/PetImages"
CATEGORIES = ["Dog", "Cat"]

for category in CATEGORIES:
    path = os.path.join(DATADIR, category)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
        plt.imshow(img_array, cmap="gray")
        plt.show()
        break
    break


# In[80]:


# Setting Up Image Scaling Section

## The images might come in different size and resolution
## Scaling the image will help to optimize our result
## The tutorial is using 50 for IMG_SIZE
## Any number upper than that wouldn't make any difference other than clearer images
## I will try 100 for IMG_SIZE to see whether we will get better results.

IMG_SIZE = 100

scale_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
plt.imshow(scale_array, cmap='gray')
plt.show()


# In[81]:


training_data = []

def creating_training_data(): 
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
                scale_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([scale_array, class_num])
            ## We must try to except the error on real dataset
            ## for simplicity purposes of the tutorial, we just passed that
            except Exception as e:
                pass

creating_training_data()


# In[82]:


print(len(training_data))


# In[83]:


import random

random.shuffle(training_data)


# In[84]:


## x = features and y = labels
## It can also be x_train & y_train and x_test & y_test for validation
## We wanted to try build in function is Keras to do validation test
## No need to seperate the array, which is neat
x = []
y = []


# In[85]:


for features, labels in training_data:
    x.append(features)
    y.append(labels)
    
## We need to convert the array to numpy array for the neural network
## 1 at the back is because the image is in grayscale
## 3 for coloured images. Will do the color version of this in the future.
x = np.array(x).reshape(-1, IMG_SIZE, IMG_SIZE, 1)


# In[86]:


import pickle

pickle_out = open("x.pickle", "wb")
pickle.dump(x, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()


# In[87]:


pickle_in = open("x.pickle", "rb")
x = pickle.load(pickle_in)


# In[88]:


x[1]

