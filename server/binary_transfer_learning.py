
import tensorflow as tf
import pandas as pd
import numpy as np
import os
from glob import glob
import random
import matplotlib.pylab as plt
import cv2
import matplotlib.gridspec as gridspec
import zlib
import itertools
import sklearn
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix
import keras
from keras.models import Sequential
from keras.optimizers import SGD, RMSprop, Adam, Adagrad, Adadelta
from keras.layers import Dense, Activation, Dropout
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Activation,Dense, Dropout, Flatten, Conv2D, MaxPool2D,MaxPooling2D,AveragePooling2D, BatchNormalization
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.models import model_from_json
from keras import backend as K
from keras.layers import Conv2D, MaxPooling2D
from keras.applications.mobilenet import MobileNet

num_epochs=4
num_class = 2 # f

X = np.load('X30k.npy')
y = np.load('y30k.npy')
df = pd.read_pickle("./df30k.pkl")

dict_characters = {0: 'Normal', 1: 'Abnormal'}

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2)
print("Training Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)
print("Training Data Shape:", len(X_train), X_train[0].shape)
print("Testing Data Shape:", len(X_test), X_test[0].shape)

Y_trainHot = to_categorical(Y_train, num_classes = num_class)
Y_testHot = to_categorical(Y_test, num_classes = num_class)

from sklearn.utils import class_weight
class_weight = class_weight.compute_class_weight('balanced', np.unique(y), y)
print(class_weight)

# Make Data 1D for compatability upsampling methods
X_trainShape = X_train.shape[1]*X_train.shape[2]*X_train.shape[3]
X_testShape = X_test.shape[1]*X_test.shape[2]*X_test.shape[3]
X_trainFlat = X_train.reshape(X_train.shape[0], X_trainShape)
X_testFlat = X_test.reshape(X_test.shape[0], X_testShape)
print("X_train Shape: ",X_train.shape)
print("X_test Shape: ",X_test.shape)
print("X_trainFlat Shape: ",X_trainFlat.shape)
print("X_testFlat Shape: ",X_testFlat.shape)

from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(ratio='auto')
X_trainRos, Y_trainRos = ros.fit_sample(X_trainFlat, Y_train)
X_testRos, Y_testRos = ros.fit_sample(X_testFlat, Y_test)
# Encode labels to hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])
Y_trainRosHot = to_categorical(Y_trainRos, num_classes =num_class )
Y_testRosHot = to_categorical(Y_testRos, num_classes = num_class)
print("X_train: ", X_train.shape)
print("X_trainFlat: ", X_trainFlat.shape)
print("X_trainRos Shape: ",X_trainRos.shape)
print("X_testRos Shape: ",X_testRos.shape)
print("Y_trainRosHot Shape: ",Y_trainRosHot.shape)
print("Y_testRosHot Shape: ",Y_testRosHot.shape)

for i in range(len(X_trainRos)):
    height, width, channels = 128,128,3
    X_trainRosReshaped = X_trainRos.reshape(len(X_trainRos),height,width,channels)
print("X_trainRos Shape: ",X_trainRos.shape)
print("X_trainRosReshaped Shape: ",X_trainRosReshaped.shape)

for i in range(len(X_testRos)):
    height, width, channels = 128,128,3
    X_testRosReshaped = X_testRos.reshape(len(X_testRos),height,width,channels)
print("X_testRos Shape: ",X_testRos.shape)
print("X_testRosReshaped Shape: ",X_testRosReshaped.shape)

# Helper Functions  Learning Curves and Confusion Matrix
from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
class MetricsCheckpoint(Callback):
    """Callback that saves metrics after each epoch"""
    def __init__(self, savepath):
        super(MetricsCheckpoint, self).__init__()
        self.savepath = savepath
        self.history = {}
    def on_epoch_end(self, epoch, logs=None):
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        np.save(self.savepath, self.history)

from sklearn.utils import class_weight
class_weight = class_weight.compute_class_weight('balanced', np.unique(y), y)
print("Old Class Weights: ",class_weight)
from sklearn.utils import class_weight
class_weight = class_weight.compute_class_weight('balanced', np.unique(Y_trainRos), Y_trainRos)
print("New Class Weights: ",class_weight)

from keras.applications.vgg16 import VGG16
from keras.models import Model
weight_path ='vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
im_size = 128
map_characters=dict_characters

map_characters={0: 'Normal', 1: 'Abnormal'}

base_model = VGG16(weights = weight_path, include_top=False, input_shape=(im_size, im_size, 3))

# Add a new top layer
x = base_model.output
x = Flatten()(x)
##predictions = Dense(num_class, activation='sigmoid')(x)

# This is the model we will train
##model = Model(inputs=base_model.input, outputs=predictions)
model = keras.models.load_model('bin_30k_weights4Ep.h5')
model.summary()

# First: train only the top layers (which were randomly initialized)
for layer in base_model.layers:
    layer.trainable = False
model.compile(loss='categorical_crossentropy', 
                  optimizer=keras.optimizers.RMSprop(lr=0.00001), 
                  metrics=['accuracy'])

callbacks_list = [keras.callbacks.EarlyStopping(monitor='val_acc', patience=3, verbose=1)]
model.summary()
history=model.fit(X_trainRosReshaped, Y_trainRosHot, epochs=num_epochs,
                  class_weight=class_weight,
                  validation_data=(X_testRosReshaped, Y_testRosHot),
                  verbose=1,callbacks = [MetricsCheckpoint('logs')])

model.save('bin_30k_weightslr00001Ep4.h5')
score = model.evaluate(X_testRosReshaped, Y_testRosHot, verbose=0)
print('\nKeras CNN #2 - accuracy:', score[1], '\n')
y_pred = model.predict(X_testRosReshaped)
print('\n', sklearn.metrics.classification_report(np.where(Y_testRosHot > 0)[1], np.argmax(y_pred, axis=1), target_names=list(map_characters.values())), sep='')
Y_pred_classes = np.argmax(y_pred,axis = 1)
Y_true = np.argmax(Y_testRosHot,axis = 1)

np.save('Y_true', Y_true)
np.save('Y_pred_classes', Y_pred_classes)
hist_acc=history.history['acc']
hist_val_acc=history.history['val_acc']
hist_loss=history.history['loss']
hist_val_loss=history.history['val_loss']
######################################
np.save('hist_acc', hist_acc)
np.savetxt('hist_acc', hist_acc)
np.save('hist_val_acc', hist_val_acc)
np.savetxt('hist_val_acc', hist_val_acc)
np.save('hist_loss', hist_loss)
np.savetxt('hist_loss', hist_loss)
np.save('hist_val_loss', hist_val_loss)
np.savetxt('hist_val_loss', hist_val_loss)

print ('All files are saved!')

