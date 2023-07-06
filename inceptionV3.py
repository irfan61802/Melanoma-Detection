# -*- coding: utf-8 -*-
"""
Created on Tue May  9 12:07:33 2023

@author: sebas
"""

import os,glob
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import pandas as pd
import cv2

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout,  GlobalAveragePooling2D
from tensorflow.keras.callbacks import Callback,EarlyStopping
from sklearn import metrics
#for ResNet50
from tensorflow.keras.applications import ResNet50, MobileNetV2
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.metrics import classification_report,confusion_matrix
#for Xception
from tf_explain.core.activations import ExtractActivations
from tensorflow.keras.applications.xception import decode_predictions
from keras.applications.xception import Xception, preprocess_input
#MobileNetV2
from tensorflow.keras.applications import MobileNetV2, InceptionV3
from tensorflow.keras import Sequential

#get file path of training data 
file_path='melanomaImages'
#get classes from folder names for benign and malignant
name_class=os.listdir(file_path)
name_class

#get and store filepaths of all images
filepaths=list(glob.glob(file_path+'/**/*.*'))
#store the labels according to folder
labels=list(map(lambda x: os.path.split(os.path.split(x)[0])[1], filepaths))

#store data as series
filepath= pd.Series(filepaths, name='Filepath').astype(str)
labels=pd.Series(labels, name='Label')
data=pd.concat([filepath, labels],axis=1)
data=data.sample(frac=1).reset_index(drop=True)
data.head(5)

#check count of each class
counts=data.Label.value_counts()
sb.barplot(x=counts.index, y=counts)
plt.xlabel('Type')
plt.xticks(rotation=90)

#train test split for validation with 0.25 
train, test= train_test_split(data, test_size=0.25, random_state=53)

#apply preprocessing: normalizing, reshaping, augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,

)
test_datagen = ImageDataGenerator(
    rescale=1./255,
)

train_gen = train_datagen.flow_from_dataframe(
    dataframe=train,
    x_col='Filepath',
    y_col='Label',
    target_size=(224,224),
    class_mode='categorical',
    batch_size=32,
    shuffle=True,
    seed=42
)
valid_gen = train_datagen.flow_from_dataframe(
    dataframe=test,
    x_col='Filepath',
    y_col='Label',
    target_size=(224,224),
    class_mode='categorical',
    batch_size=32,
    shuffle=False,
    seed=42
)
test_gen = test_datagen.flow_from_dataframe(
    dataframe=test,
    x_col='Filepath',
    y_col='Label',
    target_size=(224,224),
    class_mode='categorical',
    batch_size=32,
    shuffle=False
)

pretrained_model = InceptionV3(weights='imagenet', 
                      include_top=False,
                     input_shape=(224,224,3)
                     )


pretrained_model.trainable = False
#add a global spatial average pooling layer
x = pretrained_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(200,activation='elu')(x)
x = Dropout(0.4)(x)
x = Dense(170,activation='elu')(x)
outputs = Dense(2,activation='softmax')(x)

model = Model(inputs=pretrained_model.input, outputs=outputs)

model.compile(optimizer='adam', 
              loss='binary_crossentropy',
              metrics=['accuracy']
             )

history = model.fit(train_gen,
                    epochs=10,
                    validation_data=valid_gen)



pd.DataFrame(history.history)[['accuracy', 'val_accuracy']].plot()
plt.title("Accuracy")
plt.show()

pd.DataFrame(history.history)[['loss','val_loss']].plot()
plt.title("Loss")
plt.show()
