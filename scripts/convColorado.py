#Author: Alex Sun
#Date: 1/24/2018
#Purpose: train the nldas-grace model
#=======================================================================
import numpy as np
np.random.seed(1989)
import tensorflow as tf
tf.set_random_seed(1989)
sess = tf.Session()
import matplotlib
matplotlib.use('Agg')
import pandas as pd

from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Flatten,Activation,Reshape,Masking
from keras.layers.convolutional import Conv3D, Conv2D, MaxPooling3D, MaxPooling2D, UpSampling3D, UpSampling2D, ZeroPadding3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.optimizers import SGD, Adam,RMSprop
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, BatchNormalization, merge, TimeDistributed, LSTM #, ConvLSTM2D, UpSampling2D, merge  
from keras.layers import Convolution2D, AtrousConvolution2D
from keras.losses import categorical_crossentropy
from keras.models import load_model
import sys
import keras.backend as K
from keras import regularizers
K.set_session(sess)
from keras.utils import plot_model
from keras.applications.vgg16 import VGG16

import random
import scipy
import seaborn as sns
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from nldasColorado import GRACEData, NLDASData

'''
dim_ordering issue:
- 'th'-style dim_ordering: [batch, channels, depth, height, width]
- 'tf'-style dim_ordering: [batch, depth, height, width, channels]
'''
nb_epoch = 10    # number of epoch at training (cont) stage
batch_size = 10  # batch size

# C is number of channel
# Nf is number of antecedent frames used for training
C=1
N=64
MASKVAL=np.NaN
#for reproducibility
from numpy.random import seed
seed(1111)

def seqCNN1(seq_len=3, summary=False, backend='tf'):
    
    print("Model = seqCNN1")
    
    if backend == 'tf':
        input_shape=(N, N, seq_len) # l, h, w, c
    else:
        input_shape=(seq_len, N, N) # c, l, h, w
        
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(5, 5), input_shape=input_shape, padding='same',name='conv1',activation='relu'))
    model.add(Conv2D(32, kernel_size=(3, 3), padding='same',activation='relu',name='conv2'))
    model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu',name='conv3'))
    model.add(Conv2D(1, kernel_size=(1, 1), padding='same'))
    model.add(Activation('tanh'))
    model.add(Flatten())
    model.add(Reshape((N,N)))

    if summary:
        print(model.summary())
        #plot_model(model, to_file='cnn1model.png', show_shapes=True)

    return model

from keras.applications.vgg16 import VGG16

def seqCNNVGG16(seq_len=6, summary=False,backend='tf'):
    if backend == 'tf':
        input_shape=(N, N, seq_len) # l, h, w, c
    else:
        input_shape=(seq_len, N, N) # c, l, h, w
   
    model_vgg16_conv = VGG16(include_top=False, input_shape=input_shape, weights='imagenet')
    for layer in model_vgg16_conv.layers:
        layer.trainable = False
       
    input = Input(shape=input_shape,name = 'image_input')
    output_vgg16_conv = model_vgg16_conv(input)
    #Add the fully-connected layers
    x = Flatten(name='flatten')(output_vgg16_conv)
    x = Dense(N*N, activation='tanh', name='fc1')(x)
    x = Dropout(0.1)(x)
    x = Reshape((N,N))(x)
    model = Model(input=input, output=x)
 
    if summary:
        print(model.summary())
 
    return model

def seqCNN2(seq_len=3):
    
    print("Model = seqCNN2")
    
    input_shape=(N, N, seq_len) # l, h, w, c
        
    model = Sequential()
    model.add(Dropout(rate=0.5, input_shape=input_shape))
    model.add(Conv2D(64, kernel_size=(5, 5), padding='same',name='conv1',activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Conv2D(32, kernel_size=(3, 3), padding='same',activation='relu',name='conv2'))
    model.add(Dropout(rate=0.5))
    model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu',name='conv3'))
    model.add(Dropout(rate=0.5))
    model.add(Conv2D(1, kernel_size=(1, 1), padding='same'))
    model.add(Activation('tanh'))
    model.add(Flatten())
    model.add(Reshape((N,N)))

    print(model.summary())

    return model

def seqCNN3(seq_len=3):
    
    print("Model = seqCNN3")
    
    input_shape=(N, N, seq_len) # l, h, w, c
        
    model = Sequential()
    model.add(Dropout(rate=0.5, input_shape=input_shape))
    model.add(Conv2D(64, kernel_size=(5, 5), padding='same',name='conv1',activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Conv2D(32, kernel_size=(3, 3), padding='same',activation='relu',name='conv2'))
    model.add(Dropout(rate=0.5))
    model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu',name='conv3'))
    model.add(Dropout(rate=0.5))
    model.add(Conv2D(1, kernel_size=(1, 1), padding='same'))
    model.add(Activation('tanh'))
    model.add(Flatten())
    model.add(Reshape((N,N)))

    print(model.summary())

    return model

def lstm1(seq_len=3):
    model = Sequential()
    
    model.add(BatchNormalization(input_shape=(seq_len, N, N, 1)))
    model.add(ConvLSTM2D(filters=16, kernel_size=3, padding='same', activation='relu', return_sequences=True))
    model.add(Dropout(0.3))
    model.add(ConvLSTM2D(filters=16, kernel_size=3, padding='same',activation='relu', return_sequences=False))
    model.add(Dropout(0.3))
    
    model.add(Dense(1, activation='tanh'))
    
    model.add(Flatten())
    model.add(Reshape((N,N)))
    
    print(model.summary())
    
    return model

def lstm2(seq_len=3):
    model = Sequential()
    # define CNN model
    model.add(TimeDistributed(Conv2D(64, kernel_size=3, activation='relu'), input_shape=(N,N,seq_len,1)))
    #model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
    model.add(TimeDistributed(Flatten()))
    # define LSTM model
    model.add(LSTM(3))
    model.add(Flatten())
    model.add(Reshape((N,N)))
    print(model.summary())
    return model

def lstm3(seq_len=3):
    model = Sequential()
    model.add(LSTM(seq_len, input_shape=(N,N,seq_len, 1), return_sequences=True))
    model.add(TimeDistributed(Dense(1)))
    print(model.summary)
    return model

def lstm4(seq_len=3):
    model = Sequential()
    
    model.add(ConvLSTM2D(input_shape=(seq_len, N, N, 1), filters=20, kernel_size=3, padding='same', return_sequences=True))
    model.add(BatchNormalization())
    
    model.add(ConvLSTM2D(filters=20, kernel_size=3, padding='same', return_sequences=False))
    model.add(BatchNormalization())
    
   # model.add(Conv3D(filters=1, kernel_size=(3, 3, 3), activation='tanh', padding='same', data_format='channels_last'))
    
    model.add(Dense(1, activation='tanh'))
    
    
    model.add(Flatten())
    model.add(Reshape((N,N)))
    
    print(model.summary())
    
    return model

def uNet1(seq_len=3):
    
    print("Model = uNet1")
    
    inputs = Input((N, N, seq_len))
    #inputs = Input((ISZ, ISZ, 8))
    conv1 = Conv2D(32, 3, 3, activation='relu', border_mode='same')(inputs)
    conv1 = Conv2D(32, 3, 3, activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, 3, 3, activation='relu', border_mode='same')(pool1)
    conv2 = Conv2D(64, 3, 3, activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, 3, 3, activation='relu', border_mode='same')(pool2)
    conv3 = Conv2D(128, 3, 3, activation='relu', border_mode='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, 3, 3, activation='relu', border_mode='same')(pool3)
    conv4 = Conv2D(256, 3, 3, activation='relu', border_mode='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, 3, 3, activation='relu', border_mode='same')(pool4)
    conv5 = Conv2D(512, 3, 3, activation='relu', border_mode='same')(conv5)

    up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=3)
    conv6 = Conv2D(256, 3, 3, activation='relu', border_mode='same')(up6)
    conv6 = Conv2D(256, 3, 3, activation='relu', border_mode='same')(conv6)

    up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=3)
    conv7 = Conv2D(128, 3, 3, activation='relu', border_mode='same')(up7)
    conv7 = Conv2D(128, 3, 3, activation='relu', border_mode='same')(conv7)

    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=3)
    conv8 = Conv2D(64, 3, 3, activation='relu', border_mode='same')(up8)
    conv8 = Conv2D(64, 3, 3, activation='relu', border_mode='same')(conv8)

    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=3)
    conv9 = Conv2D(32, 3, 3, activation='relu', border_mode='same')(up9)
    conv9 = Conv2D(32, 3, 3, activation='relu', border_mode='same')(conv9)

    # For regression see: https://github.com/ncullen93/Unet-ants/blob/master/code/models/create_unet_model.py
    conv10 = Conv2D(1, 1, activation='tanh')(conv9)

    # Flatten
    flat = Flatten()(conv10)
    
    # Reshape
    reshape = Reshape((N,N))(flat)
    
    model = Model(input=inputs, output=reshape)
    
    print(model.summary())

    return model

def uNet2(seq_len=3):
    
    print("Model = uNet2")
    
    inputs = Input((N, N, seq_len))
    #inputs = Input((ISZ, ISZ, 8))
    conv1 = Conv2D(32, 3, 3, activation='relu', border_mode='same')(inputs)
    drop1 = Dropout(0.5)(conv1)
    conv1 = Conv2D(32, 3, 3, activation='relu', border_mode='same')(drop1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, 3, 3, activation='relu', border_mode='same')(pool1)
    drop2 = Dropout(0.5)(conv2)
    conv2 = Conv2D(64, 3, 3, activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, 3, 3, activation='relu', border_mode='same')(pool2)
    drop3 = Dropout(0.5)(conv3)
    conv3 = Conv2D(128, 3, 3, activation='relu', border_mode='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, 3, 3, activation='relu', border_mode='same')(pool3)
    drop4 = Dropout(0.5)(conv4)
    conv4 = Conv2D(256, 3, 3, activation='relu', border_mode='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, 3, 3, activation='relu', border_mode='same')(pool4)
    drop5 = Dropout(0.5)(conv5)
    conv5 = Conv2D(512, 3, 3, activation='relu', border_mode='same')(conv5)

    up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=3)
    conv6 = Conv2D(256, 3, 3, activation='relu', border_mode='same')(up6)
    drop6 = Dropout(0.5)(conv6)
    conv6 = Conv2D(256, 3, 3, activation='relu', border_mode='same')(conv6)

    up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=3)
    conv7 = Conv2D(128, 3, 3, activation='relu', border_mode='same')(up7)
    drop7 = Dropout(0.5)(conv7)
    conv7 = Conv2D(128, 3, 3, activation='relu', border_mode='same')(conv7)

    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=3)
    conv8 = Conv2D(64, 3, 3, activation='relu', border_mode='same')(up8)
    drop8 = Dropout(0.5)(conv8)
    conv8 = Conv2D(64, 3, 3, activation='relu', border_mode='same')(conv8)

    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=3)
    conv9 = Conv2D(32, 3, 3, activation='relu', border_mode='same')(up9)
    drop9 = Dropout(0.5)(conv9)
    conv9 = Conv2D(32, 3, 3, activation='relu', border_mode='same')(conv9)

    # For regression see: https://github.com/ncullen93/Unet-ants/blob/master/code/models/create_unet_model.py
    conv10 = Conv2D(1, 1, activation='tanh')(conv9)

    # Flatten
    flat = Flatten()(conv10)
    
    # Reshape
    reshape = Reshape((N,N))(flat)
    
    model = Model(input=inputs, output=reshape)
    
    print(model.summary())

    return model

def uNet3(seq_len=3):
    
    print("Model = uNet2")
    
    inputs = Input((N, N, seq_len))
    #inputs = Input((ISZ, ISZ, 8))
    conv1 = Conv2D(32, 3, 3, activation='relu', border_mode='same')(inputs)
    drop1 = Dropout(0.5)(conv1)
    conv1 = Conv2D(32, 3, 3, activation='relu', border_mode='same')(drop1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    batch1 = BatchNormalization()(pool1)

    conv2 = Conv2D(64, 3, 3, activation='relu', border_mode='same')(batch1)
    drop2 = Dropout(0.5)(conv2)
    conv2 = Conv2D(64, 3, 3, activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    batch2 = BatchNormalization()(pool2)

    conv3 = Conv2D(128, 3, 3, activation='relu', border_mode='same')(batch2)
    drop3 = Dropout(0.5)(conv3)
    conv3 = Conv2D(128, 3, 3, activation='relu', border_mode='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    batch3 = BatchNormalization()(pool3)

    conv4 = Conv2D(256, 3, 3, activation='relu', border_mode='same')(batch3)
    drop4 = Dropout(0.5)(conv4)
    conv4 = Conv2D(256, 3, 3, activation='relu', border_mode='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    batch4 = BatchNormalization()(pool4)

    conv5 = Conv2D(512, 3, 3, activation='relu', border_mode='same')(batch4)
    drop5 = Dropout(0.5)(conv5)
    conv5 = Conv2D(512, 3, 3, activation='relu', border_mode='same')(conv5)
    batch5 = BatchNormalization()(conv5)

    up6 = merge([UpSampling2D(size=(2, 2))(batch5), conv4], mode='concat', concat_axis=3)
    conv6 = Conv2D(256, 3, 3, activation='relu', border_mode='same')(up6)
    drop6 = Dropout(0.5)(conv6)
    conv6 = Conv2D(256, 3, 3, activation='relu', border_mode='same')(conv6)
    batch6 = BatchNormalization()(conv6)

    up7 = merge([UpSampling2D(size=(2, 2))(batch6), conv3], mode='concat', concat_axis=3)
    conv7 = Conv2D(128, 3, 3, activation='relu', border_mode='same')(up7)
    drop7 = Dropout(0.5)(conv7)
    conv7 = Conv2D(128, 3, 3, activation='relu', border_mode='same')(conv7)
    batch7 = BatchNormalization()(conv7)

    up8 = merge([UpSampling2D(size=(2, 2))(batch7), conv2], mode='concat', concat_axis=3)
    conv8 = Conv2D(64, 3, 3, activation='relu', border_mode='same')(up8)
    drop8 = Dropout(0.5)(conv8)
    conv8 = Conv2D(64, 3, 3, activation='relu', border_mode='same')(conv8)
    batch8 = BatchNormalization()(conv8)
    
    up9 = merge([UpSampling2D(size=(2, 2))(batch8), conv1], mode='concat', concat_axis=3)
    conv9 = Conv2D(32, 3, 3, activation='relu', border_mode='same')(up9)
    drop9 = Dropout(0.5)(conv9)
    conv9 = Conv2D(32, 3, 3, activation='relu', border_mode='same')(conv9)
    batch9 = BatchNormalization()(conv9)
    
    # For regression see: https://github.com/ncullen93/Unet-ants/blob/master/code/models/create_unet_model.py
    conv10 = Conv2D(1, 1, activation='tanh')(batch9)

    # Flatten
    flat = Flatten()(conv10)
    
    # Reshape
    reshape = Reshape((N,N))(flat)
    
    model = Model(input=inputs, output=reshape)
    
    print(model.summary())

    return model

def uNet4(seq_len=3):
    
    print("Model = uNet4")
    
    inputs = Input((N, N, seq_len))
    #inputs = Input((ISZ, ISZ, 8))
    conv1 = Conv2D(32, 3, 3, activation='relu', border_mode='same')(inputs)
    drop1 = Dropout(0.5)(conv1)
    conv1 = Conv2D(32, 3, 3, activation='relu', border_mode='same')(drop1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    batch1 = BatchNormalization()(pool1)

    conv2 = Conv2D(64, 3, 3, activation='relu', border_mode='same')(batch1)
    drop2 = Dropout(0.5)(conv2)
    conv2 = Conv2D(64, 3, 3, activation='relu', border_mode='same')(drop2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    batch2 = BatchNormalization()(pool2)

    conv3 = Conv2D(128, 3, 3, activation='relu', border_mode='same')(batch2)
    drop3 = Dropout(0.5)(conv3)
    conv3 = Conv2D(128, 3, 3, activation='relu', border_mode='same')(drop3)
    batch3 = BatchNormalization()(conv3)

    up4 = merge([UpSampling2D(size=(2, 2))(batch3), conv2], mode='concat', concat_axis=3)
    conv4 = Conv2D(64, 3, 3, activation='relu', border_mode='same')(up4)
    drop4 = Dropout(0.5)(conv4)
    conv4 = Conv2D(64, 3, 3, activation='relu', border_mode='same')(drop4)
    batch4 = BatchNormalization()(conv4)
    
    up5 = merge([UpSampling2D(size=(2, 2))(batch4), conv1], mode='concat', concat_axis=3)
    conv5 = Conv2D(32, 3, 3, activation='relu', border_mode='same')(up5)
    drop5 = Dropout(0.5)(conv5)
    conv5 = Conv2D(32, 3, 3, activation='relu', border_mode='same')(drop5)
    batch5 = BatchNormalization()(conv5)
    
    # For regression see: https://github.com/ncullen93/Unet-ants/blob/master/code/models/create_unet_model.py
    conv6 = Conv2D(1, 1, activation='tanh')(batch5)

    # Flatten
    flat = Flatten()(conv6)
    
    # Reshape
    reshape = Reshape((N,N))(flat)
    
    model = Model(input=inputs, output=reshape)
    
    print(model.summary())

    return model

def uNet5(seq_len=3):
    print("Model = uNet5")
    
    inputs = Input((seq_len, N, N, 1))

    size = 32
    do_rate = 0.2
    
    conv1 = ConvLSTM2D(size, 3, 3, activation='relu', border_mode='same', return_sequences=True)(inputs)
    drop1 = Dropout(do_rate)(conv1)
    conv1 = ConvLSTM2D(size, 3, 3, activation='relu', border_mode='same', return_sequences=True)(drop1)
    pool1 = MaxPooling3D(pool_size=(1, 2, 2))(conv1)

    conv2 = ConvLSTM2D(size*2, 3, 3, activation='relu', border_mode='same', return_sequences=True)(pool1)
    drop2 = Dropout(do_rate)(conv2)
    conv2 = ConvLSTM2D(size*2, 3, 3, activation='relu', border_mode='same', return_sequences=True)(drop2)
    pool2 = MaxPooling3D(pool_size=(1, 2, 2))(conv2)

    conv3 = ConvLSTM2D(size*4, 3, 3, activation='relu', border_mode='same', return_sequences=True)(pool2)
    drop3 = Dropout(do_rate)(conv3)
    conv3 = ConvLSTM2D(size*4, 3, 3, activation='relu', border_mode='same', return_sequences=True)(drop3)
    pool3 = MaxPooling3D(pool_size=(1, 2, 2))(conv3)

    conv4 = ConvLSTM2D(size*8, 3, 3, activation='relu', border_mode='same', return_sequences=True)(pool3)
    drop4 = Dropout(do_rate)(conv4)
    conv4 = ConvLSTM2D(size*8, 3, 3, activation='relu', border_mode='same', return_sequences=True)(drop4)
    pool4 = MaxPooling3D(pool_size=(1, 2, 2))(conv4)

    conv5 = ConvLSTM2D(size*16, 3, 3, activation='relu', border_mode='same', return_sequences=True)(pool4)
    drop5 = Dropout(do_rate)(conv5)
    conv5 = ConvLSTM2D(size*16, 3, 3, activation='relu', border_mode='same', return_sequences=True)(drop5)

    up6 = merge([UpSampling3D(size=(1, 2, 2))(conv5), conv4], mode='concat', concat_axis=4)
    conv6 = ConvLSTM2D(size*8, 3, 3, activation='relu', border_mode='same', return_sequences=True)(up6)
    drop6 = Dropout(do_rate)(conv6)
    conv6 = ConvLSTM2D(size*8, 3, 3, activation='relu', border_mode='same', return_sequences=True)(drop6)

    up7 = merge([UpSampling3D(size=(1, 2, 2))(conv6), conv3], mode='concat', concat_axis=4)
    conv7 = ConvLSTM2D(size*4, 3, 3, activation='relu', border_mode='same', return_sequences=True)(up7)
    drop7 = Dropout(do_rate)(conv7)
    conv7 = ConvLSTM2D(size*4, 3, 3, activation='relu', border_mode='same', return_sequences=True)(drop7)

    up8 = merge([UpSampling3D(size=(1, 2, 2))(conv7), conv2], mode='concat', concat_axis=4)
    conv8 = ConvLSTM2D(size*2, 3, 3, activation='relu', border_mode='same', return_sequences=True)(up8)
    drop8 = Dropout(do_rate)(conv8)
    conv8 = ConvLSTM2D(size*2, 3, 3, activation='relu', border_mode='same', return_sequences=True)(drop8)

    up9 = merge([UpSampling3D(size=(1, 2, 2))(conv8), conv1], mode='concat', concat_axis=4)
    conv9 = ConvLSTM2D(size, 3, 3, activation='relu', border_mode='same', return_sequences=True)(up9)
    drop9 = Dropout(do_rate)(conv9)
    conv9 = ConvLSTM2D(size, 3, 3, activation='relu', border_mode='same', return_sequences=True)(drop9)

    # For regression see: https://github.com/ncullen93/Unet-ants/blob/master/code/models/create_unet_model.py
    conv10 = ConvLSTM2D(1, 1, activation='tanh', return_sequences=False)(conv9)

    # Flatten
    flat = Flatten()(conv10)
    
    # Reshape
    reshape = Reshape((N,N))(flat)
    
    model = Model(input=inputs, output=reshape)
    
    print(model.summary())
    
    return model

def uNet6(seq_len=3):
    
    print("Model = uNet6")
    
    inputs = Input((seq_len, N, N, 1))

    size = 32
    do_rate = 0.2
    activation = 'relu'
    
    conv1 = ConvLSTM2D(size, 3, 3, activation=activation, border_mode='same', return_sequences=True)(inputs)
    drop1 = Dropout(do_rate)(conv1)
    conv1 = ConvLSTM2D(size, 3, 3, activation=activation, border_mode='same', return_sequences=True)(drop1)
    pool1 = MaxPooling3D(pool_size=(1, 2, 2))(conv1)
    batch1 = BatchNormalization()(pool1)
    
    conv2 = ConvLSTM2D(size*2, 3, 3, activation=activation, border_mode='same', return_sequences=True)(batch1)
    drop2 = Dropout(do_rate)(conv2)
    conv2 = ConvLSTM2D(size*2, 3, 3, activation=activation, border_mode='same', return_sequences=True)(drop2)
    pool2 = MaxPooling3D(pool_size=(1, 2, 2))(conv2)
    batch2 = BatchNormalization()(pool2)
    
    conv3 = ConvLSTM2D(size*4, 3, 3, activation=activation, border_mode='same', return_sequences=True)(batch2)
    drop3 = Dropout(do_rate)(conv3)
    conv3 = ConvLSTM2D(size*4, 3, 3, activation=activation, border_mode='same', return_sequences=True)(drop3)
    pool3 = MaxPooling3D(pool_size=(1, 2, 2))(conv3)
    batch3 = BatchNormalization()(pool3)
    
    conv4 = ConvLSTM2D(size*8, 3, 3, activation=activation, border_mode='same', return_sequences=True)(batch3)
    drop4 = Dropout(do_rate)(conv4)
    conv4 = ConvLSTM2D(size*8, 3, 3, activation=activation, border_mode='same', return_sequences=True)(drop4)
    pool4 = MaxPooling3D(pool_size=(1, 2, 2))(conv4)
    batch4 = BatchNormalization()(pool4)
    
    conv5 = ConvLSTM2D(size*16, 3, 3, activation=activation, border_mode='same', return_sequences=True)(batch4)
    drop5 = Dropout(do_rate)(conv5)
    conv5 = ConvLSTM2D(size*16, 3, 3, activation=activation, border_mode='same', return_sequences=True)(drop5)
    batch5 = BatchNormalization()(conv5)
    
    up6 = merge([UpSampling3D(size=(1, 2, 2))(batch5), conv4], mode='concat', concat_axis=4)
    conv6 = ConvLSTM2D(size*8, 3, 3, activation=activation, border_mode='same', return_sequences=True)(up6)
    drop6 = Dropout(do_rate)(conv6)
    conv6 = ConvLSTM2D(size*8, 3, 3, activation=activation, border_mode='same', return_sequences=True)(drop6)
    batch6 = BatchNormalization()(conv6)
    
    up7 = merge([UpSampling3D(size=(1, 2, 2))(batch6), conv3], mode='concat', concat_axis=4)
    conv7 = ConvLSTM2D(size*4, 3, 3, activation=activation, border_mode='same', return_sequences=True)(up7)
    drop7 = Dropout(do_rate)(conv7)
    conv7 = ConvLSTM2D(size*4, 3, 3, activation=activation, border_mode='same', return_sequences=True)(drop7)
    batch7 = BatchNormalization()(conv7)
    
    up8 = merge([UpSampling3D(size=(1, 2, 2))(batch7), conv2], mode='concat', concat_axis=4)
    conv8 = ConvLSTM2D(size*2, 3, 3, activation=activation, border_mode='same', return_sequences=True)(up8)
    drop8 = Dropout(do_rate)(conv8)
    conv8 = ConvLSTM2D(size*2, 3, 3, activation=activation, border_mode='same', return_sequences=True)(drop8)
    batch8 = BatchNormalization()(conv8)
    
    up9 = merge([UpSampling3D(size=(1, 2, 2))(batch8), conv1], mode='concat', concat_axis=4)
    conv9 = ConvLSTM2D(size, 3, 3, activation=activation, border_mode='same', return_sequences=True)(up9)
    drop9 = Dropout(do_rate)(conv9)
    conv9 = ConvLSTM2D(size, 3, 3, activation=activation, border_mode='same', return_sequences=True)(drop9)
    batch9 = BatchNormalization()(conv9)
    
    # For regression see: https://github.com/ncullen93/Unet-ants/blob/master/code/models/create_unet_model.py
    conv10 = ConvLSTM2D(1, 1, activation='tanh', return_sequences=False)(batch9)

    # Flatten
    flat = Flatten()(conv10)
    
    # Reshape
    reshape = Reshape((N,N))(flat)
    
    model = Model(input=inputs, output=reshape)
    
    print(model.summary())
    
    return model

def uNet7(seq_len=3):
    
    print("Model = uNet7")
    
    inputs = Input((seq_len, N, N, 1))

    size = 16
    do_rate = 0.3
    activation = 'relu'
    
    conv1 = ConvLSTM2D(size, 3, 3, activation=activation, border_mode='same', return_sequences=True)(inputs)
    batch1 = BatchNormalization()(conv1)
    conv1 = ConvLSTM2D(size, 3, 3, activation=activation, border_mode='same', return_sequences=True)(batch1)
    batch1 = BatchNormalization()(conv1)
    pool1 = MaxPooling3D(pool_size=(1, 2, 2))(batch1)
    
    
    conv2 = ConvLSTM2D(size*2, 3, 3, activation=activation, border_mode='same', return_sequences=True)(pool1)
    batch2 = BatchNormalization()(conv2)
    conv2 = ConvLSTM2D(size*2, 3, 3, activation=activation, border_mode='same', return_sequences=True)(batch2)
    batch2 = BatchNormalization()(conv2)
    pool2 = MaxPooling3D(pool_size=(1, 2, 2))(batch2)
    
    conv3 = ConvLSTM2D(size*4, 3, 3, activation=activation, border_mode='same', return_sequences=True)(pool2)
    batch3 = BatchNormalization()(conv3)
    conv3 = ConvLSTM2D(size*4, 3, 3, activation=activation, border_mode='same', return_sequences=True)(batch3)
    batch3 = BatchNormalization()(conv3)
    pool3 = MaxPooling3D(pool_size=(1, 2, 2))(batch3)
    
    conv4 = ConvLSTM2D(size*8, 3, 3, activation=activation, border_mode='same', return_sequences=True)(pool3)
    batch4 = BatchNormalization()(conv4)
    conv4 = ConvLSTM2D(size*8, 3, 3, activation=activation, border_mode='same', return_sequences=True)(batch4)
    batch4 = BatchNormalization()(conv4)
    pool4 = MaxPooling3D(pool_size=(1, 2, 2))(batch4)
    
    conv5 = ConvLSTM2D(size*16, 3, 3, activation=activation, border_mode='same', return_sequences=True)(pool4)
    batch5 = BatchNormalization()(conv5)
    conv5 = ConvLSTM2D(size*16, 3, 3, activation=activation, border_mode='same', return_sequences=True)(batch5)
    batch5 = BatchNormalization()(conv5)
    
    up6 = merge([UpSampling3D(size=(1, 2, 2))(batch5), batch4], mode='concat', concat_axis=4)
    conv6 = ConvLSTM2D(size*8, 3, 3, activation=activation, border_mode='same', return_sequences=True)(up6)
    batch6 = BatchNormalization()(conv6)
    conv6 = ConvLSTM2D(size*8, 3, 3, activation=activation, border_mode='same', return_sequences=True)(batch6)
    batch6 = BatchNormalization()(conv6)
    
    up7 = merge([UpSampling3D(size=(1, 2, 2))(batch6), batch3], mode='concat', concat_axis=4)
    conv7 = ConvLSTM2D(size*4, 3, 3, activation=activation, border_mode='same', return_sequences=True)(up7)
    batch7 = BatchNormalization()(conv7)
    conv7 = ConvLSTM2D(size*4, 3, 3, activation=activation, border_mode='same', return_sequences=True)(batch7)
    batch7 = BatchNormalization()(conv7)
    
    up8 = merge([UpSampling3D(size=(1, 2, 2))(batch7), batch2], mode='concat', concat_axis=4)
    conv8 = ConvLSTM2D(size*2, 3, 3, activation=activation, border_mode='same', return_sequences=True)(up8)
    batch8 = BatchNormalization()(conv8)
    conv8 = ConvLSTM2D(size*2, 3, 3, activation=activation, border_mode='same', return_sequences=True)(batch8)
    batch8 = BatchNormalization()(conv8)
    
    up9 = merge([UpSampling3D(size=(1, 2, 2))(batch8), batch1], mode='concat', concat_axis=4)
    conv9 = ConvLSTM2D(size, 3, 3, activation=activation, border_mode='same', return_sequences=True)(up9)
    batch9 = BatchNormalization()(conv9)
    conv9 = ConvLSTM2D(size, 3, 3, activation=activation, border_mode='same', return_sequences=True)(batch9)
    batch9 = BatchNormalization()(conv9)
    
    # For regression see: https://github.com/ncullen93/Unet-ants/blob/master/code/models/create_unet_model.py
    conv10 = ConvLSTM2D(1, 1, activation='tanh', return_sequences=False)(batch9)
    drop1 = Dropout(do_rate)(conv10)
    
    # Flatten
    flat = Flatten()(drop1)
    
    # Reshape
    reshape = Reshape((N,N))(flat)
    
    model = Model(input=inputs, output=reshape)
    
    print(model.summary())
    
    return model

def dilatedOrg(input_width=500, input_height=500):
    
    model = Sequential()
    # model.add(ZeroPadding2D((1, 1), input_shape=(input_width, input_height, 3)))
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1', input_shape=(input_width, input_height, 3)))
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))

    # Compared to the original VGG16, we skip the next 2 MaxPool layers,
    # and go ahead with dilated convolutional layers instead

    model.add(AtrousConvolution2D(512, 3, 3, atrous_rate=(2, 2), activation='relu', name='conv5_1'))
    model.add(AtrousConvolution2D(512, 3, 3, atrous_rate=(2, 2), activation='relu', name='conv5_2'))
    model.add(AtrousConvolution2D(512, 3, 3, atrous_rate=(2, 2), activation='relu', name='conv5_3'))

    # Compared to the VGG16, we replace the FC layer with a convolution

    model.add(AtrousConvolution2D(4096, 7, 7, atrous_rate=(4, 4), activation='relu', name='fc6'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(4096, 1, 1, activation='relu', name='fc7'))
    model.add(Dropout(0.5))
    # Note: this layer has linear activations, not ReLU
    model.add(Convolution2D(21, 1, 1, activation='linear', name='fc-final'))

    print(model.summary())
    # model.layers[-1].output_shape == (None, 16, 16, 21)
    return model

def dilated1(seq_len=3):
    
    print("Model = dilated1")
    
    size = 32
    
    input_shape=(N, N, seq_len)
    
    model = Sequential()
    
    # model.add(ZeroPadding2D((1, 1), input_shape=(input_width, input_height, 3)))
    model.add(Conv2D(size, kernel_size=(3, 3), activation='relu', padding='same', name='conv1_1', input_shape=input_shape))
    model.add(Conv2D(size, kernel_size=(3, 3), activation='relu', padding='same', name='conv1_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Conv2D(size*2, kernel_size=(3, 3), activation='relu', padding='same', name='conv2_1'))
    model.add(Conv2D(size*2, kernel_size=(3, 3), activation='relu', padding='same', name='conv2_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    #model.add(Conv2D(size*4, kernel_size=(3, 3), activation='relu', name='conv3_1'))
    #model.add(Conv2D(size*4, kernel_size=(3, 3), activation='relu', name='conv3_2'))
    #model.add(Conv2D(size*4, kernel_size=(3, 3), activation='relu', name='conv3_3'))
    #model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Conv2D(size*4, kernel_size=(3, 3), activation='relu', padding='same', name='conv4_1'))
    model.add(Conv2D(size*4, kernel_size=(3, 3), activation='relu', padding='same', name='conv4_2'))
    model.add(Conv2D(size*4, kernel_size=(3, 3), activation='relu', padding='same', name='conv4_3'))

    # Compared to the original VGG16, we skip the next 2 MaxPool layers,
    # and go ahead with dilated convolutional layers instead

    model.add(Conv2D(size*4, kernel_size=(3, 3), dilation_rate=(2, 2), activation='relu', padding='same', name='conv5_1'))
    model.add(Conv2D(size*4, kernel_size=(3, 3), dilation_rate=(2, 2), activation='relu', padding='same', name='conv5_2'))
    model.add(Conv2D(size*4, kernel_size=(3, 3), dilation_rate=(2, 2), activation='relu', padding='same', name='conv5_3'))

    # Compared to the VGG16, we replace the FC layer with a convolution

    #model.add(Conv2D(4096, kernel_size=(7, 7), dilation_rate=(4, 4), activation='relu', padding='valid', name='fc6'))
    #model.add(Dropout(0.5))
    model.add(Conv2D(4096, kernel_size=(1, 1), activation='relu', padding='same', name='fc7'))
    model.add(Dropout(0.5))
    model.add(UpSampling2D(size=(4, 4)))
    # Note: this layer has linear activations, not ReLU
    model.add(Conv2D(1, kernel_size=(1, 1), activation='linear', name='fc-final'))

    model.add(Flatten())
    model.add(Reshape((N,N)))
    # model.layers[-1].output_shape == (None, 16, 16, 21)
    
    print(model.summary())
    
    return model

def rmse(y_true, y_pred):
    mse= K.mean(K.square(y_pred - y_true), axis=-1)
    return mse ** 0.5

def RMSE(y_test, y_pred):
    return np.sqrt(np.mean(np.square(y_test - y_pred)))
    

def NSE(y_test, y_pred):
    term1 = np.sum(np.square(y_test - y_pred))
    term2 = np.sum(np.square(y_test - np.mean(y_test)))
    return 1.0 - term1/term2

def backTransform(nldas, mat):
    '''
    nldas, an instance of the nldas class
    '''
    temp = nldas.outScaler.inverse_transform((mat[nldas.validCells]).reshape(1, -1))
    res = np.zeros((N,N))
    res[nldas.validCells] = temp
    return res

def backTransformIn(nldas, mat):
    '''
    nldas, an instance of the nldas class
    '''
    temp = nldas.inScaler.inverse_transform((mat[nldas.validCells]).reshape(1, -1))
    res = np.zeros((N,N))
    res[nldas.validCells] = temp
    return res

def plotOutput(nldas, ypred, Y_test, X_test):
    #plot image
    cmap = ListedColormap((sns.color_palette("RdBu", 10)).as_hex())
    for i in range(0, ypred.shape[0],10):
        plt.figure() 
        plt.subplot(1,2,1)       
        obj = backTransformIn(nldas, X_test[i,:,:,0])
        obj[nldas.mask==0.0] = np.NaN
        vmin=-12.5
        vmax=12.5
        im=plt.imshow(obj, cmap, origin='lower')
        plt.colorbar(im, orientation="horizontal", fraction=0.046, pad=0.1)
        plt.clim(vmin,vmax)
        plt.subplot(1,2,2)
        obj = backTransformIn(nldas, X_test[i,:,:,0])-backTransform(nldas, ypred[i,:,:])
        obj[nldas.mask==0.0] = np.NaN               
        im=plt.imshow(obj, cmap, origin='lower')
        plt.colorbar(im, orientation="horizontal", fraction=0.046, pad=0.1)   
        plt.clim(vmin,vmax)     
        plt.savefig('testout%s.png'%i)

def calculateBasinAverage(nldas, Y):
    '''
    Y is either predicted or test tensor
    '''
    nvalidCell = nldas.nvalidCells
    tws_avg = np.zeros((Y.shape[0]))
    for i in range(Y.shape[0]):
        obj = backTransform(nldas, Y[i,:,:])
        tws_avg[i] = np.sum(obj)/nvalidCell
    return tws_avg

def calculateInAverage(nldas, Y):
    nvalidCell = nldas.nvalidCells
    tws_avg = np.zeros((Y.shape[0]))
    for i in range(Y.shape[0]):
        obj = nldas.inScaler.inverse_transform(np.reshape(Y[i,:,:],(1,N*N)))
        tws_avg[i] = np.sum(obj)/nvalidCell
    return tws_avg


    
def reshapeLSTMX(X, seq_len=3):
    '''
    Format training data to work with LSTM
    X: (num_months, X, Y, seq_len)
    out: (num_months, seq_len, X, Y, 1)
    '''
    out = np.transpose(X, (0, 3, 1, 2))
    return np.reshape(out, (-1, seq_len, N, N, 1))

def augment(X, Y, seq_len, n=1000):
    '''
    Creates augumented samples by randomly adding small pertubations to values
    '''
    num_samples = np.shape(X)[0]
    newX = np.zeros((n, 64, 64, 3))
    newY = np.zeros((n, 64, 64))
    for i in range(n):

        sampleidx = random.choice(range(num_samples))
        augX = X[sampleidx,:,:,:]       
        xidx = random.choice(range(64))    
        yidx = random.choice(range(64))     
        valueidx = random.choice(range(3))    
        delta = np.random.uniform(np.mean(X), np.random.uniform(low=-0.1, high=0.1))     
        
        #val = X[sampleidx, xidx, yidx, valueidx] + delta  
        #augX[xidx, yidx, valueidx] = val
        
        vals = X[sampleidx, xidx, yidx, valueidx] + delta 
        augX[xidx, yidx] = vals
        
        newX[i] = augX   
        newY[i] = Y[sampleidx]
        
    
    # Finally concat the original and augmented sets
    newX = np.append(X, newX)
    newY = np.append(Y, newY)
    
    return newX, newY

def fitAugmented(x_train, y_train, model, epochs=100):
    datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

    # compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied)
    datagen.fit(x_train)

    # fits the model on batches with real-time data augmentation:
    model.fit_generator(datagen.flow(x_train, y_train, batch_size=32),
                        steps_per_epoch=len(x_train) / 32, epochs=epochs)
    
def correlationSummary(corrections, train_end_idx=104, test_end_idx=140):
    '''
    Computes correlation statistics for a given models predictions
    '''
    
    # 2 = ignore first 2 values ???
    observations = pd.read_csv('observations.txt', delim_whitespace=True)[2:]
    corrections = corrections[:test_end_idx]
    
    nldas_obs = observations['nldas'][:test_end_idx]
    nldas_train = nldas_obs[:train_end_idx] 
    nldas_test = nldas_obs[train_end_idx:test_end_idx]
    
    grace_obs = observations['grace'][:test_end_idx]
    grace_train = grace_obs[:train_end_idx]
    grace_test = grace_obs[train_end_idx:test_end_idx]
    
    baseline_total_corr,_ = scipy.stats.pearsonr(nldas_obs, grace_obs)
    baseline_train_corr,_ = scipy.stats.pearsonr(nldas_train, grace_train)
    baseline_test_corr,_ = scipy.stats.pearsonr(nldas_test, grace_test)
    
    print('Baseline Total Corr=%s' % baseline_total_corr)
    print('Baseline Train Corr=%s' % baseline_train_corr)
    print('Baseline Test Corr=%s' % baseline_test_corr)
    
    corrected = np.subtract(nldas_obs, corrections)
    corrected_train = corrected[:train_end_idx]
    corrected_test = corrected[train_end_idx:test_end_idx]
    
    model_total_corr,_ = scipy.stats.pearsonr(corrected, grace_obs)
    model_train_corr,_ = scipy.stats.pearsonr(corrected_train, grace_train)
    model_test_corr,_ = scipy.stats.pearsonr(corrected_test, grace_test)
    
    print('Model Total Corr=%s' % model_total_corr)
    print('Model Train Corr=%s' % model_train_corr)
    print('Model Test Corr=%s' % model_test_corr)
    
    plt.plot(nldas_train, '-', grace_train, '.-', nldas_test, '.', grace_test, '--')
    
    
def CNNDriver(watershedName='colorado', retrain=False):
    grace = GRACEData(watershed=watershedName)    
    nldas = NLDASData(watershed=watershedName)
    nldas.loadStudyData(reloadData=False)    

    n_p = 3
    model = seqCNN1(seq_len=n_p)
    #model = seqCNNVGG16(seq_len=n_p)
    #model = uNet7(seq_len=n_p)
    #model = lstm1(seq_len=n_p)
    
    X_train,Y_train,X_test,Y_test,Xval = nldas.formMatrix2D(gl=grace, n_p=n_p, masking=True, nTrain=106)
    
    # Transform for use with LSTM models
    #X_train,X_test = [reshapeLSTMX(x, seq_len=n_p) for x in (X_train,X_test)]
    
    # Augment
    #X_train,Y_train = augment(X_train, Y_train)
    
    solver=1
    if retrain:     
        solver=1
        if solver==1:   
            lr = 1e-3  # learning rate
            solver = Adam(lr=lr)    
        elif solver==2:
            lr = 1e-4
            solver = RMSprop(lr=lr, decay=0.9)
        model.compile(loss='mse', optimizer=solver, metrics=[rmse])
     
        model.fit(X_train, Y_train, epochs=nb_epoch, batch_size=batch_size, shuffle=False, verbose=1)
        #fitAugmented(X_train, Y_train, model)
        
        model.save_weights('{0}model_weights.h5'.format(watershedName))
    else:
        model.load_weights('{0}model_weights.h5'.format(watershedName))
    
    mse = np.zeros((Y_test.shape[0]))
    ypred = model.predict(X_test, batch_size=batch_size, verbose=0)
    print(ypred.shape)
    for i in range(ypred.shape[0]):        
        mse[i]=RMSE(Y_test[i,:,:], ypred[i,:,:])
        
    print('RMSE=%s' % np.mean(mse))
    
    '''
    nse = np.zeros((Y_test.shape[0]))
    for i in range(ypred.shape[0]):        
        nse[i]=NSE(Y_test[i,:,:], ypred[i,:,:])
        
    print('NSE=%s' % np.mean(nse))
    '''
    
    #comment out the following to calculate basin average time series
    
    ytrain = model.predict(X_train, batch_size=batch_size, verbose=0)
    twsTrain = calculateBasinAverage(nldas, ytrain)    
    #for item in twsTrain:
    #    print(item)
    
    twsPred = calculateBasinAverage(nldas, ypred)    
    
    correlationSummary(np.concatenate([twsTrain, twsPred]))
    
    #twsIn = calculateBasinAverage(nldas, Xval)
    #twsnldasavg = nldas.twsnldas[-Y_test.shape[0]:]   
    
    #twsgraceTrain = nldas.twsgrace[-Y_train.shape[0]:]
    #twsgrace = nldas.twsgrace[-Y_test.shape[0]:]
    #print('predicted shape', twsPred.shape)
    #print('test shape', twsgrace.shape)
    #for i in range(len(twsPred)):
    #    print('%s\t%s\t%s' % (twsnldasavg[i], twsgrace[i], twsPred[i]))
    
    #for i in range(len(twsPred)):
    #    print('%s' % (twsPred[i]))
        
    #plotOutput(nldas, ypred, Y_test, X_test)

def main(retrain=False, modelnum=1):
    watershed='colorado'
    if modelnum==1:
        CNNDriver(watershedName=watershed, retrain=retrain)
        
if __name__ == "__main__":
    main(retrain=True, modelnum=1)