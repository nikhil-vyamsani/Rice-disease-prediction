#!/usr/bin/env python
# coding: utf-8

# In[62]:


# Import Libraries
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Keras API
import keras
from keras.utils import load_img
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten
from keras.layers import Conv2D,MaxPooling2D,Activation,AveragePooling2D,BatchNormalization
from keras.preprocessing.image import ImageDataGenerator

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


# In[63]:


# Loading train and test data into seperate variables
train_dir =r"C:\Users\V NIKHIL KUMAR\Desktop\AI Project\archive\LabelledRice\Labelled"
test_dir=r"C:\Users\V NIKHIL KUMAR\Desktop\AI Project\archive\RiceDiseaseDataset\validation"


# In[64]:


#returns a arbitary list containing the names of the entries in the directory given by path
os.listdir(r"C:\Users\V NIKHIL KUMAR\Desktop\AI Project\archive\RiceDiseaseDataset\validation")


# In[65]:


def get_files(directory):
  if not os.path.exists(directory): #if path doesn't exist
    return 0
  count=0
  for current_path,dirs,files in os.walk(directory): #walk() method generates the file names in a directory tree by walking the tree either top-down or bottom-up
    for dr in dirs:
      count+= len(glob.glob(os.path.join(current_path,dr+"/*"))) #counts number of files in path by joining currenpath with path given at runtime
  return count


# In[66]:


train_samples =get_files(train_dir)        #returns no.of images in train data
num_classes=len(glob.glob(train_dir+"/*")) #returns no.of classes in train data
test_samples=get_files(test_dir)           #returns no.of images in test data
print(num_classes,"Classes")
print(train_samples,"Train images")
print(test_samples,"Test images")


# In[67]:


# Pre-processing images with ImageDataGenerator function parameters
train_datagen=ImageDataGenerator(rescale=1./255,        #transform every pixel value from range [0,255] -> [0,1]
                                   shear_range=0.2,     #randomly applying shearing transformations
                                   zoom_range=0.2,      #randomly zooming inside pictures
                                   rotation_range=40,
                                   horizontal_flip=True,
                                   vertical_flip=True)#randomly flipping half of the images horizontally
test_datagen=ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,    
                                   zoom_range=0.2,   
                                   rotation_range=40,
                                   horizontal_flip=True,
                                   vertical_flip=True)       
#whatever preprocessing done to train images same is done to test images


# In[68]:


#Generating augmented data from train and test directories
img_width,img_height =256,256        #set height and width and color of input image  
input_shape=(img_width,img_height,3)
batch_size =32                       #refers to the number of training examples utilized in one iteration
train_generator =train_datagen.flow_from_directory(train_dir, #Takes the path to directory,and generates batches of augmented data
                                                   target_size=(img_width,img_height),                                                    
                                                   batch_size=batch_size)
test_generator=test_datagen.flow_from_directory(test_dir,
                                                shuffle=True,
                                                target_size=(img_width,img_height),
                                                batch_size=batch_size)


# In[69]:


# returns names of classes
train_generator.class_indices


# In[75]:


# CNN Model building
model = Sequential()
model.add(Conv2D(64, (5, 5),input_shape=input_shape,activation='relu'))
model.add(Conv2D(64, (5, 5),activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Conv2D(64, (3, 3),activation='relu'))
model.add(Conv2D(64, (3, 3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3),activation='relu'))
model.add(Conv2D(128, (3, 3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))   
model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(512,activation='relu'))          
model.add(Dense(num_classes,activation='softmax'))
model.summary()


# In[76]:


from tensorflow.keras.preprocessing import image
import numpy as np
img1 = image.load_img(r'C:\Users\V NIKHIL KUMAR\Desktop\AI Project\archive\RiceDiseaseDataset\validation\Hispa\\IMG_20190419_094308.jpg')
plt.imshow(img1);
#preprocessed image sample
img1 = image.load_img(r'C:\Users\V NIKHIL KUMAR\Desktop\AI Project\archive\RiceDiseaseDataset\validation\Hispa\\IMG_20190419_094308.jpg', target_size=(256, 256))
img = image.img_to_array(img1)
img = img/255
img = np.expand_dims(img, axis=0)


# In[77]:


#Start training CNN with parameters
#Generating validation augmented data from train directories
validation_generator = test_datagen.flow_from_directory(
                       test_dir, # same directory as training data
                       target_size=(img_height, img_width),
                       batch_size=batch_size)
opt=keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])


# In[ ]:


train=model.fit(train_generator,steps_per_epoch=15, epochs=10, verbose=1, callbacks=None,
    validation_data=validation_generator, validation_steps=None, validation_freq=1,
    class_weight=None, max_queue_size=10, workers=1, use_multiprocessing=False,
    shuffle=True, initial_epoch=0
)


# In[74]:


print(max(train.history['accuracy']))


# In[26]:


# Save model
from keras.models import load_model
model.save('crop.h1')


# In[27]:


from keras.models import load_model
model=load_model('crop.h1')
# Mention name of the disease into list.
Classes = ['BrownSpot', 'Healthy', 'Hispa', 'LeafBlast']


# In[61]:


import numpy as np
import matplotlib.pyplot as plt
from keras.utils import load_img
from tensorflow.keras.preprocessing import image 
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
# Pre-Processing test data same as train data.
img_width=256
img_height=256
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

def prepare(img_path):
    img = load_img(img_path, target_size=(256, 256))
    x = image.img_to_array(img)
    x = x/255
    return np.expand_dims(x, axis=0)

predict_x=model.predict(prepare(r'C:\Users\V NIKHIL KUMAR\Desktop\AI Project\archive\RiceDiseaseDataset\Test\BrownSpot\\IMG_2992.jpg')) 
classes_x=np.argmax(predict_x,axis=1)
disease=image.load_img(r'C:\Users\V NIKHIL KUMAR\Desktop\AI Project\archive\RiceDiseaseDataset\Test\BrownSpot\\IMG_2992.jpg')
plt.imshow(predict_x)
x = Classes[int(classes_x)]
print("plant is/has :",x)
l1 = predict_x.tolist()


# In[ ]:




