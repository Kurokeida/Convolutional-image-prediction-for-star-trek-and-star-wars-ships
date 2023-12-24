import os 
import cv2
import re
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy import asarray
import tensorflow as tf
from tensorflow import image
from tensorflow.keras.utils import to_categorical as TC
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications import imagenet_utils
from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import OneHotEncoder as OHE
from sklearn.compose import ColumnTransformer as CT

#%matplotlib notebook
#%matplotlib inline
/
FilePath = (r"")
string_literal = '\\'

ImageFolderNames = os.listdir(FilePath)
training_images = []
float_images = []
testing_values = []
classes = []

for temp_appends in ImageFolderNames:
    classes.append(temp_appends)

temp_lists = []

for x1 in os.listdir(FilePath):
    temp_lists.append(f"{FilePath}{string_literal}{x1}")
    
len_num = len(temp_lists)
print(len_num)

fullImagePaths = []

for x2 in temp_lists:
    for x3 in os.listdir(x2):
        fullImagePaths.append(f"{x2}{string_literal}{x3}")


cv_paths = []
for mm in fullImagePaths:
    pil_image1=Image.open(mm)
    pil_image2 = np.array(pil_image1)
    pil_image3 = cv2.cvtColor(pil_image2, cv2.COLOR_BGR2RGB)
    pil_image4 = cv2.resize(pil_image3,(32,32), interpolation = cv2.INTER_AREA)
    cv_paths.append(pil_image4)

cv_paths = np.array(cv_paths)
cv_paths1 = cv_paths/255

print(type(cv_paths1))
print(cv_paths1.shape)
plt.imshow(cv_paths1[55])
print(cv_paths1[55])

new_training = np.asarray(cv_paths1)
print(new_training)

batch_size = new_training.shape[0]
width = new_training.shape[1]
height = new_training.shape[2]
color_channels = 3

new_training = new_training.reshape(batch_size,width,height,color_channels)
print(new_training.shape)
print(new_training)

small_test_values = []
for x4 in temp_lists:
    small_test_values.append(x4[49:])

for i55 in small_test_values:
    print(i55)

raw_test_values = []

removable_chars = []
for x5 in range(10):
    removable_chars.append(str(x5))

print(removable_chars)

for x7 in temp_lists:
    for x8 in os.listdir(x7):
        raw_test_values.append(x8)

nuked_testing_values = []

for x6 in raw_test_values:
    rx = '[' + re.escape(''.join(removable_chars)) + ']'
    new_x4 = re.sub(rx,'',x6)
    nuked_testing_values.append(new_x4[:-4])

Second_testing_values = []
for temp_item in nuked_testing_values:
    
    if temp_item == "BirdOfPlay":
        Second_testing_values.append(0)
    if temp_item == "Deridex":
        Second_testing_values.append(1)
    if temp_item == "DroughtNought":
        Second_testing_values.append(2)
    if temp_item == "DS":
        Second_testing_values.append(3)
    if temp_item == "EmperialShuttle":
        Second_testing_values.append(4)
    if temp_item == "EnterpriseD":
        Second_testing_values.append(5)
    if temp_item == "StarDestroyer":
        Second_testing_values.append(6)
    if temp_item == "TieFighter":
        Second_testing_values.append(7)
    if temp_item == "X-Wing":
        Second_testing_values.append(8)
    if temp_item == "Dderix":
        Second_testing_values.append(9)

print(Second_testing_values)

final_testing_set = TC(Second_testing_values)

for im in final_testing_set:
    print(im)

for cc in cv_paths1:
    print(cc)

print(color_channels)

model = Sequential()

model.add(Conv2D(
    filters=32,
    kernel_size=(4,4),
    input_shape=(32,32,3),
    activation='relu'
))
model.add(MaxPooling2D(
    pool_size=(2,2)
))

model.add(Conv2D(
    filters=64,
    kernel_size=(4,4),
    input_shape=(32,32,3),
    activation='relu'
))
model.add(MaxPooling2D(
    pool_size=(2,2)
))

model.add(Conv2D(
    filters=32,
    kernel_size=(4,4),
    input_shape=(32,32,3),
    activation='relu'
))
model.add(MaxPooling2D(
    pool_size=(2,2)
))


model.add(Flatten())

model.add(Dense(256,activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(10,activation='softmax'))

model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy']
             )

early_stop = EarlyStopping(monitor='val_loss',patience=1)

print(final_testing_set)

for uuu in new_training:
    print(uuu)

print(final_testing_set)

model.fit(new_training,
          final_testing_set,
          epochs=25,
          callbacks=[early_stop],
          validation_data=(new_training,final_testing_set)
         )

metrics = pd.DataFrame(model.history.history)

metrics[['loss','val_loss']].plot()
metrics[["accuracy","val_accuracy"]].plot()

trainingInfo = model.evaluate(new_training,final_testing_set)
print("Loss:",trainingInfo[0])
print("Accuracy:",trainingInfo[1])

print(prediction_path6.shape)
#prediction_path6 = prediction_path4.reshape(None,32,32,3)
print(prediction_path6)

plt.imshow(prediction_path1)

pre1 = np.asarray(prediction_path6)
print(type(prediction_path6))
print(type(pre1))

prediction = model.predict(pre1)
index = np.argmax(prediction)

print(prediction)
print(index)
print(small_test_values[index])
