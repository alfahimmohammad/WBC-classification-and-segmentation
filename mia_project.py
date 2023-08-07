# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 21:28:47 2021

@author: alfah
"""

#%%
#import the libraries and dataset
import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten, \
  Conv2D, BatchNormalization, ZeroPadding2D, MaxPooling2D, Activation, add
from tensorflow.keras.models import Model
from tensorflow.keras import Sequential
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import mixed_precision
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger

from sklearn.metrics import multilabel_confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

from glob import glob
#mixed_precision.set_global_policy('mixed_float16')

IMAGE_SIZE = [224, 224] # original: 240, 320feel free to change depending on dataset

# training config:
epochs = 30
batch_size = 10

# https://www.kaggle.com/paultimothymooney/blood-cells
train_path = 'blood_cell_images/TRAIN'
valid_path = 'blood_cell_images/TEST'

# useful for getting number of files
image_files = glob(train_path + '/*/*.jp*g')
valid_image_files = glob(valid_path + '/*/*.jp*g')

# look at an image for fun
plt.imshow(image.load_img(np.random.choice(image_files)))
plt.show()
#run one of the following cells with respect to whicever model you want to initialize, train\test. 
#%%
#custom resnet
def identity_block(input_, kernel_size, filters):
    f1, f2, f3 = filters

    x = Conv2D(f1, (1, 1), kernel_initializer='he_normal')(input_)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(f2, kernel_size, padding='same',kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(f3, (1, 1), kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)

    x = add([x, input_])
    x = Activation('relu')(x)
    return x

def conv_block(input_, kernel_size, filters, strides=(2, 2)):
    f1, f2, f3 = filters

    x = Conv2D(f1, (1, 1), strides=strides, kernel_initializer='he_normal')(input_)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(f2, kernel_size, padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(f3, (1, 1), kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)

    shortcut = Conv2D(f3, (1, 1), strides=strides, kernel_initializer='he_normal')(input_)
    shortcut = BatchNormalization()(shortcut)

    x = add([x, shortcut])
    x = Activation('relu')(x)
    return x

# our custom resnet
input = Input(shape=IMAGE_SIZE + [3])
x = ZeroPadding2D(padding=(3, 3))(input)

x = Conv2D(64, (7, 7), strides=(2, 2), padding='valid', kernel_initializer='he_normal')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

x = ZeroPadding2D(padding=(1, 1))(x)

x = MaxPooling2D((3, 3), strides=(2, 2))(x)

x = conv_block(x, 3, [64, 64, 256], strides=(1, 1))
x = identity_block(x, 3, [64, 64, 256])
x = identity_block(x, 3, [64, 64, 256])

x = conv_block(x, 3, [128, 128, 512])
x = identity_block(x, 3, [128, 128, 512])
x = identity_block(x, 3, [128, 128, 512])
x = identity_block(x, 3, [128, 128, 512])

# our layers - you can add more if you want
x = Flatten()(x)
# x = Dense(1000, activation='relu')(x)
prediction = Dense(4, activation='softmax')(x)

# create a model object
model = Model(inputs=input, outputs=prediction)
# view the structure of the model
model.summary()
#%%
#vgg16
model = Sequential()
model.add(Conv2D(input_shape=(224,224,1),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
#model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
#model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
#model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Flatten())
model.add(Dense(units=4096,activation="relu"))
model.add(Dense(units=2048,activation="relu"))
model.add(Dense(units=4, activation="softmax"))

model.summary()
#%%
#xception
model = tf.keras.applications.Xception(
    include_top=True,
    weights=None,
    input_tensor=None,
    input_shape=IMAGE_SIZE + [3],
    pooling=None,
    classes=4,
    classifier_activation="softmax",
    )
model.summary()
#%%
#inception_resnet
model = tf.keras.applications.InceptionResNetV2(
    include_top=True,
    weights=None,
    input_tensor=None,
    input_shape=IMAGE_SIZE + [3],
    pooling=None,
    classes=4,
    classifier_activation="softmax",    
    )
model.summary()
#%%
#nasnet_large
model = tf.keras.applications.NASNetLarge(
    input_shape=IMAGE_SIZE + [3],
    include_top=True,
    weights=None,
    input_tensor=None,
    pooling=None,
    classes=4,
    )
model.summary()
#%%
#resnet101v2
model = tf.keras.applications.ResNet101V2(
    include_top=True,
    weights=None,
    input_tensor=None,
    input_shape=IMAGE_SIZE + [3],
    pooling=None,
    classes=4,
    classifier_activation="softmax",
    )
model.summary()
#%%
# tell the model what cost and optimization method to use
#change saved_model_dir and model name depending on your task
auc = tf.keras.metrics.AUC(multi_label = True)
saved_model_dir = 'C:/Users/alfah/OneDrive/Desktop/fahim/IITM/1st_sem/Medical_Image_Analysis/Term_paper_project/inc_res_rgb_checkpoints'
model.load_weights(saved_model_dir+'/inc_res_rgb_model-014-0.307892_0.985041.hdf5')
model.compile(
  loss='categorical_crossentropy',
  optimizer=Adam(learning_rate=0.0001),
  metrics=['accuracy', auc]
)
color_mode = 'rgb'
# create an instance of ImageDataGenerator
def preprocess_input2(x):
  x /= 127.5
  x -= 1.
  return x

train_gen = ImageDataGenerator(
  rotation_range=20,
  width_shift_range=0.1,
  height_shift_range=0.1,
  shear_range=0.1,
  zoom_range=0.2,
  horizontal_flip=True,
  vertical_flip=True,
  preprocessing_function=preprocess_input2
)

val_gen = ImageDataGenerator(
  preprocessing_function=preprocess_input2
)

# test generator to see how it works and some other useful things

# get label mapping for confusion matrix plot later
test_gen = val_gen.flow_from_directory(valid_path, 
                                       target_size=IMAGE_SIZE, 
                                       shuffle = False, 
                                       class_mode='categorical', 
                                       color_mode=color_mode, 
                                       batch_size=1)
print(test_gen.class_indices)
labels = [None] * len(test_gen.class_indices)
for k, v in test_gen.class_indices.items():
  labels[v] = k
  
# should be NOT a strangely colored image
for x, y in test_gen:
  print("min:", x[0].min(), "max:", x[0].max())
  plt.title(labels[np.argmax(y[0])])
  plt.imshow(x[0,:,:,:], cmap='gray')
  plt.show()
  break

#%%
# create generators
train_generator = train_gen.flow_from_directory(
  train_path,
  target_size=IMAGE_SIZE,
  shuffle=True,
  batch_size=batch_size,
  class_mode='categorical',
  color_mode=color_mode
)
valid_generator = val_gen.flow_from_directory(
  valid_path,
  target_size=IMAGE_SIZE,
  shuffle=True,
  batch_size=batch_size,
  class_mode='categorical',
  color_mode=color_mode
)
#%%
#to check if the output dimesions of image and labels match our expectations
for i, (x,y) in enumerate(test_gen):
    print(x.shape, y.shape)
    if i==0:
        break
#%%
# fit the model
#os.mkdir('/Checkpoints')
#csv_looger to store training history
#change model names in the csv logger and save_path in the model_checkpoint callback depending on which model you want to train
csv_logger = CSVLogger(saved_model_dir+'/resnet101v2_rgb_training_history.csv', append=True, separator=',')

r = model.fit(
  train_generator,
  validation_data=valid_generator,
  epochs=epochs,
  initial_epoch = 0,
  steps_per_epoch=len(image_files) // batch_size,
  validation_steps=len(valid_image_files) // batch_size,
  callbacks=[
      csv_logger,
      ModelCheckpoint(
      filepath=saved_model_dir+'/resnet101v2_rgb_model-{epoch:03d}-{val_loss:03f}_{val_auc:03f}.hdf5',
      save_weights_only=True,
      monitor='val_auc',
      mode='max',
      save_best_only=True),
  ]
)
"""
    tf.keras.callbacks.EarlyStopping(
      monitor='loss', patience=3, restore_best_weights=True),
"""
#%%
#testing
model.evaluate(test_gen, 
               batch_size = batch_size)
#%%
#calculating the performance metrics
batch_size = 1
thresh = 0.5
y_preds = np.zeros((len(valid_image_files), 4), dtype = 'float32')
y_trues = np.zeros((len(valid_image_files), 4), dtype = 'float32')
Y_preds = np.zeros((len(valid_image_files), 4), dtype = 'float32')
Y_trues = np.zeros((len(valid_image_files), 4), dtype = 'float32')
l=0

for i, (x, z) in enumerate(test_gen):
    if i>=2487:
        break
    y_true = z
    batch_size = y_true.shape[0]
    y_pred = model.predict([x])
    y_preds[l:l+batch_size,:] = y_pred
    Y_preds[l:l+batch_size,:] = y_pred
    y_trues[l:l+batch_size,:] = y_true
    l += batch_size

Y_preds[Y_preds>=thresh] = 1.0
Y_preds[Y_preds<thresh] = 0.0

conf_matrix = multilabel_confusion_matrix(y_trues, Y_preds)
print(conf_matrix)

auc1 = tf.keras.metrics.AUC(multi_label = False)
for j in range(0,4):
    print("Class: ", j)
    c_matrix = conf_matrix[j,:,:]
    print('Sensitivity: ', (c_matrix[1, 1]/(c_matrix[1, 1]+c_matrix[1, 0]))*100)
    print('Specificity: ', (c_matrix[0, 0]/(c_matrix[0, 0]+c_matrix[0, 1]))*100)
    print('Precision: ', (c_matrix[1, 1]/(c_matrix[1, 1]+c_matrix[0, 1]))*100)
    print('Accuracy: ', (c_matrix[0, 0]+c_matrix[1, 1])/(np.sum(c_matrix))*100)
    auc1.reset_states()
    auc1.update_state(y_trues[:,j], y_preds[:,j])
    print('AUC: ',auc1.result().numpy())