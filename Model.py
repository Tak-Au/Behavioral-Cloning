import pandas as pd 
from enum import Enum
import cv2
import numpy as np


datafile = "/home/tak/CarND-Term1-Starter-Kit/Projects/Project3/CarND-Behavioral-Cloning-P3/Data/driving_log.csv"

df = pd.read_csv(datafile)

samplefile = df['center'][0]
img = cv2.imread(samplefile)
imageshape = img.shape




def HShiftandBrightess(img, istrainning=True):
    if (istrainning):
        offsetfactor = 10 #parameter
        brightnessvalue = 25

        n = np.random.randint(-offsetfactor,offsetfactor) 
        resultimg = np.zeros_like(img)
        width = resultimg.shape[1]
        if (n>0):
            resultimg[:,:(width-n)] = img[:,n:]
        else:
            resultimg[:,-n:] = img[:,:(width+n)] 

        # Convert BGR to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        hsv[:,:,2] =  hsv[:,:,2] + int(np.random.randint(-brightnessvalue,brightnessvalue))
        resultimg = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    else:
        resultimg = img
    return resultimg


import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

KTF.set_session(tf.Session(config=config))


from sklearn.model_selection import train_test_split
import sklearn

batch_size = 64



def generator(df, batch_size=32, istrainning = True):
    num_samples = len(df)
    filenameprefix = ''
                
    while 1:
        correction = 0.27
        for offset in range(0, num_samples,batch_size):
            batchdatas = df.iloc[offset:offset+batch_size]
            X_data= []
            y_data = []
            for img_center, img_left, img_right,steering_center in zip (batchdatas['center'],batchdatas['left'],batchdatas['right'], batchdatas['steering']):
                steering_center = float(steering_center)
                img_center = filenameprefix+img_center
                X_data.append(HShiftandBrightess(cv2.imread(img_center),istrainning))
                y_data.append(steering_center)
                X_data.append(np.fliplr(HShiftandBrightess(cv2.imread(img_center),istrainning)))
                y_data.append(-steering_center)   
                img_left = (filenameprefix+img_left).replace(" ","")
                img_right = filenameprefix+img_right.replace(" ","")

                steering_left = steering_center + correction
                steering_right = steering_center - correction
                X_data.append(HShiftandBrightess(cv2.imread(img_left),istrainning))
                y_data.append(steering_left)  
                X_data.append(HShiftandBrightess(cv2.imread(img_right),istrainning))
                y_data.append(steering_right) 
            X_data = np.array(X_data)
            y_data = np.array(y_data)
            yield sklearn.utils.shuffle(X_data, y_data)

train_samples, validation_samples = train_test_split(df, test_size=0.2)


train_generator = generator(train_samples, batch_size)
validation_generator = generator(validation_samples, batch_size, False)


from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout,Reshape
from keras.layers import MaxPooling2D, Flatten, BatchNormalization
from keras.layers.convolutional import Convolution2D
from keras import optimizers
import keras
from keras.models import load_model
from keras.layers import Lambda
from keras.layers import Cropping2D
from keras.callbacks import ModelCheckpoint


epochs = 10

model = Sequential()
model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(imageshape)))
model.add(Lambda(lambda x: (x / 255.0) - 0.5))
model.add(Convolution2D(24, 5, 5, subsample=(2,2)))
#model.add(BatchNormalization())
model.add(Activation('elu'))
#model.add(MaxPooling2D())
model.add(Convolution2D(36, 5, 5, subsample=(2,2)))
#model.add(BatchNormalization())
model.add(Activation('elu'))
#model.add(MaxPooling2D())
model.add(Convolution2D(48, 5, 5, subsample=(2,2)))
#model.add(BatchNormalization())
model.add(Activation('elu'))
#model.add(MaxPooling2D())
model.add(Convolution2D(64, 3, 3))
#model.add(BatchNormalization())
model.add(Activation('relu'))
#model.add(MaxPooling2D())
model.add(Convolution2D(64, 3, 3))
#model.add(BatchNormalization())
model.add(Activation('elu'))
#model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(1164))
model.add(Dropout(.50))
model.add(Dense(100))
model.add(Dropout(.50))
model.add(Dense(50))
model.add(Dropout(.50))
model.add(Dense(10))
model.add(Dense(1))

model.summary()


model.compile(loss='mse',
              optimizer='adam', learningrate = 0.0005)


filepath = 'NvidiaModel.{epoch:02d}.h5'
saveModel = ModelCheckpoint(filepath, monitor='val_loss',
                            verbose=0, save_best_only=False, 
                            save_weights_only=False,
                            mode='auto', period=1)

saveBestModel = ModelCheckpoint('NvidiaBestModel.h5', monitor='val_loss',
                            verbose=0, save_best_only=True, 
                            save_weights_only=False,
                            mode='auto', period=1)


model.fit_generator(generator = train_generator,
          samples_per_epoch= len(train_samples)*4,
          validation_data = validation_generator,
          nb_val_samples= len(validation_samples)*4,
          nb_epoch=epochs, callbacks = [saveModel, saveBestModel])