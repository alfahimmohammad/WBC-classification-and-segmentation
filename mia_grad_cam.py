# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 10:50:50 2021

@author: alfah
"""

#%%

import tensorflow as tf
from tensorflow import keras

# Display
from IPython.display import Image, display
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from tensorflow.keras.layers import Input, Lambda, Dense, Flatten, \
  Conv2D, BatchNormalization, ZeroPadding2D, MaxPooling2D, Activation, add
from tensorflow.keras.models import Model
from tensorflow.keras import Sequential
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import mixed_precision
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger

from sklearn.metrics import confusion_matrix
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
last_conv_layer_name = "block14_sepconv2_act"
saved_model_dir = 'C:/Users/alfah/OneDrive/Desktop/fahim/IITM/1st_sem/Medical_Image_Analysis/Term_paper_project/xception_rgb_checkpoints'
color_mode = 'rgb'
model.load_weights(saved_model_dir+'/xception_rgb_model-003-0.347696_0.986925.hdf5')
model.layers[-1].activation = None

# create an instance of ImageDataGenerator
def preprocess_input2(x):
  x /= 127.5
  x -= 1.
  return x

val_gen = ImageDataGenerator(
  preprocessing_function=preprocess_input2
)

test_gen = val_gen.flow_from_directory(valid_path, target_size=IMAGE_SIZE, class_mode='categorical', color_mode=color_mode)
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
valid_generator = val_gen.flow_from_directory(
  valid_path,
  target_size=IMAGE_SIZE,
  shuffle=False,
  batch_size=1,
  class_mode='categorical',
  color_mode=color_mode
)

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_display_gradcam(x, heatmap, cam_path="cam.jpg", alpha=0.4):
    # Load the original image
    #img = keras.preprocessing.image.load_img(img_path)
    #img = keras.preprocessing.image.img_to_array(img)
    img = (x + 1.) * 127.5
    img = img.astype('float32')
    #

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

    # Save the superimposed image
    superimposed_img.save(cam_path)

    # Display Grad CAM
    display(Image(cam_path))
#%%
label = 0
for i, (x,y) in enumerate(valid_generator):
    if y[0,label] == 1:
        preds = model.predict(x)
        pred = np.squeeze(preds.numpy())
        if np.argmax(pred) == label:
            heatmap = make_gradcam_heatmap(x.numpy(), model, last_conv_layer_name, pred_index = label)
            save_and_display_gradcam(x, heatmap)
            break
            
            