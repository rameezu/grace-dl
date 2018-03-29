#Author: Alex Sun
#Date: 1/24/2018
#Purpose: train the gldas-grace model
#date: 2/16/2018 adapt for the gldas
#date: 3/7/2018 adapt for global
#date: 3/8/2018 adapt for local bigger area study
#=======================================================================
import numpy as np
np.random.seed(1989)
import tensorflow as tf
tf.set_random_seed(1989)
sess = tf.Session()
import matplotlib
matplotlib.use('Agg')

from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Flatten,Activation,Reshape,Masking
from keras.layers.convolutional import Conv3D, Conv2D, MaxPooling3D, MaxPooling2D, ZeroPadding3D

from keras.optimizers import SGD, Adam,RMSprop
import keras
from keras.layers import Input, BatchNormalization,ConvLSTM2D, UpSampling2D  
from keras.losses import categorical_crossentropy
from keras.models import load_model
import sys
import keras.backend as K
from keras import regularizers
import seaborn as sns
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from gldasLocal import GRACEData, GLDASData, NDVIData
import gldasLocal
from keras.callbacks import ReduceLROnPlateau

K.set_session(sess)
from keras.utils import plot_model

from scipy.stats.stats import pearsonr
import pandas as pd
from ProcessIndiaDB import ProcessIndiaDB
from numpy.random import seed
seed(1111)
import pickle as pkl
from statsmodels.distributions.empirical_distribution import ECDF

'''
dim_ordering issue:
- 'th'-style dim_ordering: [batch, channels, depth, height, width]
- 'tf'-style dim_ordering: [batch, depth, height, width, channels]
'''
nb_epoch = 30   # number of epoch at training (cont) stage
batch_size = 15  # batch size

# C is number of channel
# Nf is number of antecedent frames used for training
C=1
N=gldasLocal.N
MASKVAL=np.NaN
reduce_lr = ReduceLROnPlateau(monitor='rmse', factor=0.5,
              patience=3, min_lr=0.000001)

def getLSTMLayer(inputLayerName, Ns, seq_len=3, backend='tf'):
    '''
    Ns=size of input image, assuming square
    seq_len= # of samples
    '''
    if backend == 'tf':
        input_shape=(seq_len, Ns, Ns, 1) # samples, times, h, w, c
    else:
        input_shape=(seq_len, 1, Ns, Ns) # samples, times, c, h, w
    
    inputLayer = Input(shape=input_shape, name=inputLayerName)
        
    layer = ConvLSTM2D(32, kernel_size=(3,3), padding="same",  kernel_regularizer=regularizers.l2(0.01), bias_initializer='zeros')(inputLayer)
    layer = BatchNormalization()(layer)
    layer = ConvLSTM2D(16, kernel_size=(3,3), padding="same", return_sequences=True)(layer)
    layer = BatchNormalization()(layer)
    layer = ConvLSTM2D(16, kernel_size=(3,3), padding="same", return_sequences=True)(layer)
    layer = BatchNormalization()(layer)    
    #layer = Conv2D(32, kernel_size=(3,3), padding="same",  bias_initializer='zeros', activation="relu")(layer)    
    #layer = Conv2D(16, kernel_size=(3,3), padding="same",  bias_initializer='zeros', activation="relu")(layer)
    #layer = BatchNormalization()(layer)
    return layer, inputLayer

def seqCNN1(seq_len=6, summary=False,backend='tf'):
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
        plot_model(model, to_file='cnn1model.png', show_shapes=True)

    return model

def seqCNN3(seq_len=3, summary=False,backend='tf'):
    if backend == 'tf':
        input_shape=(N, N, seq_len) # l, h, w, c
    else:
        input_shape=(seq_len, N, N) # c, l, h, w
        
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), input_shape=input_shape, padding='same',name='conv1',activation='relu'))
    model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, kernel_size=(3, 3), padding='same',activation='relu'))
    model.add(Conv2D(64, kernel_size=(3, 3), padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu'))  
    model.add(UpSampling2D(size=(8, 8)))
    
    model.add(Conv2D(1, kernel_size=(1, 1), padding='same', activation='tanh'))
    model.add(Flatten())
    model.add(Reshape((N,N)))


    if summary:
        print(model.summary())
        plot_model(model, to_file='seqCNN3model.png', show_shapes=True)
    return model

from keras.applications.vgg16 import VGG16
def vcg16CNN(seq_len=6, summary=False,backend='tf'):
    '''
    the transfer learning model
    '''
    if backend == 'tf':
        input_shape=(N, N, seq_len) # l, h, w, c
    else:
        input_shape=(seq_len, N, N) # c, l, h, w
    
    model_vgg16_conv = VGG16(include_top=False, input_shape=input_shape, weights='imagenet')
    plot_model(model_vgg16_conv, to_file='vgg16model2.png', show_shapes=True)
    for layer in model_vgg16_conv.layers:
        layer.trainable = False
        
    input = Input(shape=input_shape,name = 'image_input')
    output_vgg16_conv = model_vgg16_conv(input)
    #Add the fully-connected layers 
    x = Flatten(name='flatten')(output_vgg16_conv)
    x = Dropout(0.2)(x)    
    x = Dense(N*N, activation='tanh', name='fc1')(x)
    x = Dropout(0.2)(x)
    x = Reshape((N,N))(x)
    model = Model(input=input, output=x)

    if summary:
        print(model.summary())
        plot_model(model, to_file='vgg16model1.png', show_shapes=True)
    return model

from keras.applications.resnet50 import ResNet50
def resnetCNN(seq_len=6, summary=False, backend='tf'):
    '''
    the resnet transfer learning model
    '''
    if backend == 'tf':
        input_shape=(N, N, seq_len) # l, h, w, c
    else:
        input_shape=(seq_len, N, N) # c, l, h, w
    
    modelresnet = ResNet50(include_top=False, input_shape=input_shape, weights='imagenet')
    for layer in modelresnet.layers:
        layer.trainable = False
    
    input = Input(shape=input_shape,name = 'image_input')
    output_conv = modelresnet(input)
    #Add the fully-connected layers 
    x = Flatten(name='flatten')(output_conv)
    x = Dropout(0.2)(x)    
    x = Dense(N*N, activation='tanh', name='fc1')(x)
    x = Dropout(0.2)(x)
    x = Reshape((N,N))(x)
    model = Model(input=input, output=x)

    if summary:
        print(model.summary())
    return model

def getUnetModel(Ns, seq_len=3, summary=False):

    inputs = Input((N, N, seq_len))
    conv1 = Conv2D(32, kernel_size=(3, 3), activation='relu', border_mode='same')(inputs)
    conv1 = Conv2D(32, kernel_size=(3, 3), activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, kernel_size=(3, 3), activation='relu', border_mode='same')(pool1)
    conv2 = Conv2D(64, kernel_size=(3, 3), activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, kernel_size=(3, 3), activation='relu', border_mode='same')(pool2)
    conv3 = Conv2D(128, kernel_size=(3, 3), activation='relu', border_mode='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, kernel_size=(3, 3), activation='relu', border_mode='same')(pool3)
    conv4 = Conv2D(256, kernel_size=(3, 3), activation='relu', border_mode='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, kernel_size=(3, 3), activation='relu', border_mode='same')(pool4)
    conv5 = Conv2D(512, kernel_size=(3, 3), activation='relu', border_mode='same')(conv5)

    up6 = keras.layers.concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=-1)
    conv6 = Conv2D(256, kernel_size=(3, 3), activation='relu', border_mode='same')(up6)
    conv6 = Conv2D(256, kernel_size=(3, 3), activation='relu', border_mode='same')(conv6)

    up7 = keras.layers.concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=-1)
    conv7 = Conv2D(128, kernel_size=(3, 3), activation='relu', border_mode='same')(up7)
    conv7 = Conv2D(128, kernel_size=(3, 3), activation='relu', border_mode='same')(conv7)

    up8 = keras.layers.concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=-1)
    conv8 = Conv2D(64, kernel_size=(3, 3), activation='relu', border_mode='same')(up8)
    conv8 = Conv2D(64, kernel_size=(3, 3), activation='relu', border_mode='same')(conv8)

    up9 = keras.layers.concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=-1)
    conv9 = Conv2D(32, kernel_size=(3, 3), activation='relu', border_mode='same')(up9)
    conv9 = Conv2D(32, kernel_size=(3, 3), activation='relu', border_mode='same')(conv9)

    conv10 = Conv2D(1, kernel_size=(1, 1), activation='tanh')(conv9)
    x = Flatten()(conv10)
    x = Reshape((N,N))(x)
    model = Model(input=inputs, output=x)
    #model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=[jaccard_coef, jaccard_coef_int, 'accuracy'])
    if summary:
        print(model.summary())
    return model    

def getLSTMModel(Ns, seq_len=3, backend='tf'):
    if backend == 'tf':
        input_shape=(seq_len, Ns, Ns, 1) # samples, times, h, w, c
    else:
        input_shape=(seq_len, 1, Ns, Ns) # samples, times, c, h, w    
    seq = Sequential()
    seq.add(ConvLSTM2D(filters=64, kernel_size=(3, 3),
                       input_shape=input_shape,
                       padding='same', return_sequences=True))
    seq.add(BatchNormalization())
    
    seq.add(ConvLSTM2D(filters=32, kernel_size=(3, 3),
                       padding='same', return_sequences=True))
    seq.add(BatchNormalization())
    
    seq.add(ConvLSTM2D(filters=32, kernel_size=(3, 3),
                       padding='same', return_sequences=True))
    seq.add(BatchNormalization())
    
    seq.add(ConvLSTM2D(filters=32, kernel_size=(3, 3),
                       padding='same', return_sequences=False))
    seq.add(BatchNormalization())
    
    seq.add(Conv2D(filters=1, kernel_size=(3,3),
                   activation='tanh',
                   padding='same', data_format='channels_last'))
    seq.add(Flatten())
    seq.add(Reshape((N,N)))   
    return seq

def compositeLSTM(n_p=None, summary=False, backend='tf'):
    if n_p is None:
        n_p = 3
    '''
    x, main_input = getLSTMLayer(inputLayerName='main_input', Ns=N, seq_len=n_p)
    #x = MaxPooling2D(pool_size=(3,3))(cnnlayer)
    #x = Conv2D(32, kernel_size=(3,3), padding='same', activation='relu')(x)
    #x = MaxPooling2D(pool_size=(3,3))(x)
    x = Conv2D(1, kernel_size=(1,1), padding='same', activation='relu')(x)
    #x = Flatten()(x)    
    #x = Dense(N*N, activation='tanh')(x)
    main_output = Reshape((N,N))(x)
    
    model = Model(inputs=main_input, outputs=[main_output])
    '''
    model = getLSTMModel(Ns=N, seq_len=n_p)
    if summary:
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

    
def LSTMDriver(watershedName, watershedInner, retrain=False):
    grace = GRACEData(reLoad=False, watershed=watershedName)    
    nldas = GLDASData(watershed=watershedName,watershedInner=watershedInner)
    nldas.loadStudyData(reloadData=False)    

    n_p = 3
    model = compositeLSTM(n_p=n_p, summary=True)
    X_train,Y_train,X_test,Y_test = nldas.formMatrix2DLSTM(gl=grace, n_p=n_p, masking=True, nTrain=106)

    solver=1
    if retrain:     
        solver=1
        if solver==1:   
            lr = 1e-2  # learning rate
            solver = Adam(lr=lr)    
        elif solver==2:
            lr = 1e-4
            solver = RMSprop(lr=lr, decay=0.9)
        model.compile(loss='mse', optimizer=solver, metrics=[rmse])
        
        history = model.fit(X_train, Y_train,
                            epochs=nb_epoch,
                            batch_size=batch_size,
                            shuffle=True,
                            verbose=1)
        model.save_weights('glo{0}lstmmodel_weights.h5'.format(watershedName))
    else:
        model.load_weights('glo{0}lstmmodel_weights.h5'.format(watershedName))
    
    mse = np.zeros((Y_test.shape[0]))
    ypred = model.predict(X_test, batch_size=batch_size, verbose=0)
    print ypred.shape
    for i in range(ypred.shape[0]):        
        mse[i]=RMSE(Y_test[i,:,:], ypred[i,:,:])

    print 'RMSE=%s' % np.mean(mse)

def CNNCompositeDriver(watershedName, watershedInner, n_p=3, nTrain=125, retrain=False):
    '''
    This  used  precipitation data
    '''
    isMasking=True
    grace = GRACEData(reLoad=False, watershed=watershedName)    
    nldas = GLDASData(watershed=watershedName,watershedInner=watershedInner)
    nldas.loadStudyData(reloadData=False, masking=isMasking)    

    X_train,Y_train,X_test,Y_test,_ = nldas.formMatrix2D(gl=grace, n_p=n_p, masking=isMasking, nTrain=nTrain)

    Xp_train, Xp_test = nldas.formPrecip2D(n_p=n_p, masking=isMasking, nTrain=nTrain)

    backend='tf'
    if backend == 'tf':
        input_shape=(N, N, n_p) # l, h, w, c
    else:
        input_shape=(n_p, N, N) # c, l, h, w
    
    inLayer = Input(shape=input_shape, name='input')
    outLayer = Conv2D(64, kernel_size=(3, 3), padding='same',activation='relu')(inLayer)
    
    inputPLayer = Input(shape=(N,N,n_p), name='inputP')
    outPLayer = Conv2D(64, kernel_size=(3, 3), padding='same',activation='relu')(inputPLayer)
                                                                                    
    x = keras.layers.concatenate([outPLayer, outLayer], axis=-1)            
    x = Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    x = Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    x = Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu')(x)    
    x = UpSampling2D(size=(8, 8))(x)
    
    x = Conv2D(1, kernel_size=(1, 1), padding='same', activation='tanh')(x)
    x = Flatten()(x)

    x = Reshape((N,N))(x)    
    
    model = Model(inputs=[inputPLayer, inLayer], outputs=[x])
    print model.summary()
    plot_model(model, to_file='cnncompositemodel.png', show_shapes=True)

    solver=3
    if retrain:     
        solver=1
        if solver==1:   
            lr = 1e-3  # learning rate
            solver = Adam(lr=lr)    
        elif solver==2:
            lr = 1e-4
            solver = RMSprop(lr=lr, decay=0.9)
        elif solver==3:
            lr = 1e-2
            solver = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
                        
        model.compile(loss='mse', optimizer=solver, metrics=[rmse])
        
        history = model.fit([Xp_train, X_train], Y_train,
                            epochs=nb_epoch,
                            batch_size=batch_size,
                            shuffle=True,
                            verbose=1, callbacks=[reduce_lr])
        model.save_weights('glo{0}compomodel_weights.h5'.format(watershedName))
        np.save('{0}history.npy'.format('comp'), history.history)
    else:
        model.load_weights('glo{0}compmodel_weights.h5'.format(watershedName))

    doTesting(model, 'comp', nldas, X_train, X_test, Y_test, Xp_train, Xp_test, n_p=n_p, nTrain=nTrain)

def CNN1(inputLayer):
    '''
    for use with composite model
    '''
    x = Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu')(inputLayer)
    x = Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    x = Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu')(x)    
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu')(x)    
    x = UpSampling2D(size=(8, 8))(x)
        
    x = Conv2D(1, kernel_size=(1, 1), padding='same', activation='tanh')(x)
    x = Flatten()(x)

    x = Reshape((N,N))(x)    
    
    return x

def CNNvgg16(inputLayer):
    '''
    the transfer learning model for use with composite model
    '''
    input_shape=[N, N, inputLayer._keras_shape[3]] # l, h, w, c
    print 'input shape = ', input_shape
    print inputLayer._keras_shape

    model_vgg16_conv = VGG16(include_top=False, input_shape=input_shape, weights='imagenet')
    for layer in model_vgg16_conv.layers:
        layer.trainable = False
        
    output_vgg16_conv = model_vgg16_conv(inputLayer)
    #Add the fully-connected layers 
    x = Flatten(name='flatten')(output_vgg16_conv)
    x = Dropout(0.2)(x)    
    x = Dense(N*N, activation='tanh', name='fc1')(x)
    x = Dropout(0.2)(x)
    x = Reshape((N,N))(x)
    
    return x


def CNNNDVICompositeDriver(watershedName, watershedInner, n_p=3, nTrain=125, modelOption=N, retrain=False):
    '''
    This  used  precipitation data
    '''
    isMasking=True
    grace = GRACEData(reLoad=False, watershed=watershedName)    
    nldas = GLDASData(watershed=watershedName,watershedInner=watershedInner)
    nldas.loadStudyData(reloadData=False,masking=isMasking)    
    ndvi = NDVIData(reLoad=False, watershed=watershedName) 
    
    X_train,Y_train,X_test,Y_test,_ = nldas.formMatrix2D(gl=grace, n_p=n_p, masking=isMasking, nTrain=nTrain)
    Xp_train, Xp_test = nldas.formPrecip2D(n_p=n_p, masking=isMasking, nTrain=nTrain)
    Xnd_train, Xnd_test = nldas.formNDVI2D(ndvi, n_p=n_p, masking=isMasking, nTrain=nTrain)

    backend='tf'
    if backend == 'tf':
        input_shape=(N, N, n_p) # l, h, w, c
    else:
        input_shape=(n_p, N, N) # c, l, h, w


    ioption=modelOption
    if ioption==1:
        inLayer = Input(shape=input_shape, name='input')
        outLayer = Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu')(inLayer)
            
        inputPLayer = Input(shape=(N,N,n_p), name='inputP')
        outPLayer = Conv2D(64, kernel_size=(3, 3), padding='same',activation='relu')(inputPLayer)
    
        inputNDVILayer = Input(shape=(N,N,n_p), name='inputNDVI')
        outNDVILayer = Conv2D(64, kernel_size=(3, 3), padding='same',activation='relu')(inputNDVILayer)

        x = keras.layers.concatenate([outPLayer, outNDVILayer, outLayer], axis=-1)            
        x = CNN1(x)
        label = 'ndvicnn'
    elif ioption==2:
        inLayer = Input(shape=input_shape, name='input')
        outLayer = Conv2D(1, kernel_size=(3, 3), padding='same', activation='relu')(inLayer)
            
        inputPLayer = Input(shape=(N,N,n_p), name='inputP')
        outPLayer = Conv2D(1, kernel_size=(3, 3), padding='same',activation='relu')(inputPLayer)
    
        inputNDVILayer = Input(shape=(N,N,n_p), name='inputNDVI')
        outNDVILayer = Conv2D(1, kernel_size=(3, 3), padding='same',activation='relu')(inputNDVILayer)
        x = keras.layers.concatenate([outPLayer, outNDVILayer, outLayer], axis=-1)            

        x = CNNvgg16(x)
        
        label = 'ndvivgg'
    
                                                                                                        
    model = Model(inputs=[inputPLayer, inputNDVILayer, inLayer], outputs=[x])
    print(model.summary())
    plot_model(model, to_file='{0}model.png'.format(label), show_shapes=True)
    
    solver=3
    if retrain:     
        solver=1
        if solver==1:   
            lr = 1e-3  # learning rate
            solver = Adam(lr=lr)    
        elif solver==2:
            lr = 1e-4
            solver = RMSprop(lr=lr, decay=0.9)
        elif solver==3:
            lr = 1e-2
            solver = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
        
        model.compile(loss='mse', optimizer=solver, metrics=[rmse])
        
        history = model.fit([Xp_train, Xnd_train, X_train], Y_train,
                            epochs=nb_epoch,
                            batch_size=batch_size,
                            shuffle=True, callbacks=[reduce_lr],
                            verbose=1)
        model.save_weights('glo{0}ndvimodel_weights{1}.h5'.format(watershedName,label))
        np.save('{0}history.npy'.format(label), history.history)
    else:
        model.load_weights('glo{0}ndvimodel_weights{1}.h5'.format(watershedName,label))

    doTesting(model, label, nldas, X_train, X_test, Y_test, Xp_train, Xp_test,
              Xnd_train, Xnd_test, 
              n_p=n_p, nTrain=nTrain, pixel=False)
    

def CNNDriver(watershedName, watershedInner, n_p=3, nTrain=125, retrain=False, modelOption=1):
    isMasking=True
    grace = GRACEData(reLoad=False, watershed=watershedName)    
    nldas = GLDASData(watershed=watershedName,watershedInner=watershedInner)
    nldas.loadStudyData(reloadData=False, masking=isMasking)
    
    ioption= modelOption
    
    if ioption==1:
        model = vcg16CNN(seq_len=n_p, summary=True)
        label='vgg16' #should be vgg16
    elif ioption==2:
        model = seqCNN3(n_p, summary=True)
        label = 'simple'
        
    X_train,Y_train,X_test,Y_test,Xval = nldas.formMatrix2D(gl=grace, n_p=n_p, 
                                                            masking=isMasking, nTrain=nTrain)
    
    solver=3
    if retrain:     
        solver=1
        if solver==1:   
            lr = 1e-3  # learning rate
            solver = Adam(lr=lr)    
        elif solver==2:
            lr = 1e-4
            solver = RMSprop(lr=lr, decay=0.9)
        elif solver==3:
            lr = 1e-2
            solver = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
            
        model.compile(loss='mse', optimizer=solver, metrics=[rmse])
        
        history = model.fit(X_train, Y_train,
                            epochs=nb_epoch,
                            batch_size=batch_size,
                            shuffle=True,
                            verbose=1, callbacks=[reduce_lr])
        model.save_weights('glo{0}model_weights{1}.h5'.format(watershedName, ioption))
        np.save('{0}history.npy'.format(label), history.history)
    else:
        model.load_weights('glo{0}model_weights{1}.h5'.format(watershedName, ioption))
        
    doTesting(model, label, nldas, X_train, X_test, Y_test, n_p=n_p, nTrain=nTrain, pixel=True) 
    
def UnetDriver(watershedName, watershedInner, retrain=False):
    grace = GRACEData(reLoad=False, watershed=watershedName)    
    nldas = GLDASData(watershed=watershedName,watershedInner=watershedInner)
    nldas.loadStudyData(reloadData=False)    

    n_p = 3
    nTrain=106
    model = getUnetModel(Ns=N, seq_len=3, summary=True)
    X_train,Y_train,X_test,Y_test,Xval = nldas.formMatrix2D(gl=grace, n_p=n_p, masking=True, nTrain=nTrain)
    
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
        
        history = model.fit(X_train, Y_train,
                            epochs=nb_epoch,
                            batch_size=batch_size,
                            shuffle=True,
                            verbose=1)
        model.save_weights('{0}unetmodel_weights.h5'.format(watershedName))
    else:
        model.load_weights('{0}unetmodel_weights.h5'.format(watershedName))
    
    doTesting(model, 'unet', nldas, X_train, X_test, Y_test, n_p=n_p, nTrain=nTrain)
  
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

def calcRMat(gldas,  Ytrain, Ypred, Xtrain, Xpred, n_p, validationOnly=False):
    def getMat(Yin, Xin):
        mat = np.zeros((Yin.shape[0], gldas.nvalidCells), dtype=np.float64)
        for i in range(Yin.shape[0]):
            obj = backTransformIn(gldas, Xin[i,:,:,0])-backTransform(gldas, Yin[i,:,:])
            mat[i,:] = obj[gldas.validCells]
        return mat
    if not validationOnly:
        #correlation over the whole period
        matTrain = getMat(Ytrain, Xtrain)
        matPred = getMat(Ypred, Xpred)
        #concatenate along the rows
        mat = np.r_[matTrain,matPred[1:]]
        corrmat = np.zeros((gldas.nvalidCells), dtype='float64')
        nMonths = mat.shape[0]
        print 'array dimensions', mat.shape[0], gldas.graceArr.shape[0]
        for i in range(mat.shape[1]):
            corrmat[i],_ = pearsonr(mat[:,i], gldas.graceArr[n_p-1:nMonths+n_p,i])
    else:
        #correlation over the validation period only
        matPred = getMat(Ypred, Xpred)
        #concatenate along the rows
        mat = matPred
        corrmat = np.zeros((gldas.nvalidCells), dtype='float64')
        nTrain = Ytrain.shape[0]
        print 'array dimensions', mat.shape[0], gldas.graceArr.shape[0]
        for i in range(mat.shape[1]):
            corrmat[i],_ = pearsonr(mat[:,i], gldas.graceArr[nTrain:-1,i])
        
    #convert to image
    temp = np.zeros((N,N))+np.NaN
    temp[gldas.validCells] = corrmat

    return temp

def plotCorrmat(label, gldas, corrected=None, masking=True):
    '''
    plot cnn vs. grace correlation grid map 
    '''
    #plot image
    cmap = ListedColormap((sns.diverging_palette(240, 10, n=9)).as_hex())
    
    fig=plt.figure(figsize=(12,6),dpi=300) 
    plt.subplot(1,3,1)
    pp = np.zeros(gldas.mask.shape)+np.NaN
    pp[gldas.mask==1] = 1.0
    ppActual = np.zeros(gldas.innermask.shape)+np.NaN
    #masking for india itself
    ppActual[gldas.innermask==1]=1.0

    if masking:
        vmin=-0.8; vmax=1.0        
        temp = np.multiply(gldas.gldas_grace_R, pp)
        im=plt.imshow(temp, cmap, origin='lower', alpha=0.9, interpolation='bilinear', vmin=vmin,vmax=vmax)
        #temp = np.multiply(gldas.gldas_grace_R, ppActual)
        #im=plt.imshow(temp, cmap, origin='lower', alpha=0.95, interpolation='bilinear',vmin=vmin,vmax=vmax)      
    else:
        im=plt.imshow(gldas.gldas_grace_R, cmap, origin='lower')
        
    cx,cy = gldas.getBasinBoundForPlotting()

    plt.plot(cx,cy, '-', color='#7B7D7D')
    plt.colorbar(im, orientation="horizontal", fraction=0.046, pad=0.1)
    plt.title('$R_{GLDAS/GRACE}$')
    
    if not corrected is None:
        if masking:
            vmin=-0.4; vmax=1.0
            corrected = np.multiply(corrected, pp)
            correctedInner = np.multiply(corrected, ppActual)            
            gldas_grace_R_inner=np.multiply(gldas.gldas_grace_R, ppActual)
            #save for CDF plot
            np.save('corrdist{0}.npy'.format(label), [correctedInner[ppActual==1],gldas_grace_R_inner[ppActual==1]])
        #compute correlation between GRACE and learned values
        plt.subplot(1,3,2)            
        im=plt.imshow(corrected, cmap, origin='lower', alpha=0.9, interpolation='bilinear',vmin=vmin,vmax=vmax)
        #im=plt.imshow(correctedInner, cmap, origin='lower', alpha=0.95, interpolation='bilinear',vmin=vmin,vmax=vmax)
        plt.colorbar(im, orientation="horizontal", fraction=0.046, pad=0.1)
        plt.plot(cx,cy, '-', color='#7B7D7D')
        plt.title('$R_{DL/GRACE}$')
        
        plt.subplot(1,3,3)
        vmin=-0.6;vmax=0.6
        dd = corrected-gldas.gldas_grace_R
        im=plt.imshow(dd, cmap, origin='lower',alpha=0.9, interpolation='bilinear',vmin=vmin,vmax=vmax)
        dd = np.multiply(dd, ppActual)
        #im=plt.imshow(dd, cmap, origin='lower',alpha=0.9, interpolation='bilinear',vmin=vmin,vmax=vmax)    
        plt.colorbar(im, orientation="horizontal", fraction=0.046, pad=0.1)
        plt.plot(cx,cy, '-', color='#7B7D7D')
        plt.clim(-.6, 0.6)
        plt.title('$\Delta R$')
        print np.nanmin(corrected-gldas.gldas_grace_R)
        print np.nanmax(corrected-gldas.gldas_grace_R)
    plt.savefig('cnn_grace_corr%s.png' % label, dpi=fig.dpi,transparent=True, frameon=False)
        
def calculateBasinAverage(nldas, Y):
    '''
    Y is either predicted or test tensor
    '''
    nvalidCell = nldas.nActualCells
    tws_avg = np.zeros((Y.shape[0]))
    for i in range(Y.shape[0]):
        obj = backTransform(nldas, np.multiply(Y[i,:,:], nldas.innermask))
        tws_avg[i] = np.sum(obj)/nvalidCell
    return tws_avg

def calculateInAverage(nldas, Y):
    '''
    '''
    nvalidCell = nldas.nActualCells
    tws_avg = np.zeros((Y.shape[0]))
    for i in range(Y.shape[0]):
        obj = nldas.inScaler.inverse_transform(np.reshape(Y[i,:,:],(1,N*N)))
        tws_avg[i] = np.sum(obj)/nvalidCell
    return tws_avg

def calculateBasinPixel(nldas, Y, iy, ix):
    tws_ts = np.zeros((Y.shape[0]))
    for i in range(Y.shape[0]):
        obj = backTransform(nldas, Y[i,:,:])
        tws_ts[i] = obj[iy,ix]
    return tws_ts

def calcCNN_GW_RMat(gldas, Ytrain, Ypred, n_p):
    '''
    calculate correlation between in situ gw anomaly and predicted
    '''

    def getMat(Yin):
        mat = np.zeros((Yin.shape[0], len(gldas.gwvalidcells[0])), dtype=np.float64)
        for i in range(Yin.shape[0]):
            obj = -backTransform(gldas, Yin[i,:,:])
            mat[i,:] = obj[gldas.gwvalidcells]
        return mat

    matTrain = getMat(Ytrain)
    matPred = getMat(Ypred)
    #contatenate along the rows
    mat = np.r_[matTrain,matPred[1:]]
    #extract records corresponding to gw measurements
    #gldas mat time axis is from 2002/04, gw starts from 2005/01
    rng = np.array([33,37,40,43], dtype='int')
    b = np.array([], dtype='int')
    for i in range(2005,2014):
        b = np.r_[b, rng]
        rng = rng+12
    print 'corr indices', b
    #need to shift to account for n_p
    #mat = mat[b-(n_p-1),:]
    mat = mat[b-(n_p-1),:]
    gwmat = gldas.gwanomalyAvgDF.as_matrix() 
    
    corrmat = np.zeros((mat.shape[1]), dtype='float64')

    for i in range(mat.shape[1]):
        #only do correlation on non-nan cells
        nas = np.isnan(gwmat[:,i])        
        corrmat[i],_ = pearsonr(mat[~nas,i], gwmat[~nas,i])
    
    #convert to image
    temp = np.zeros((N,N))+np.NaN
    temp[gldas.gwvalidcells] = corrmat
    print 'mean gw corr is', np.nanmean(corrmat)
    return temp

def plotGwCorrmat(label, gldas, gwcorrmat, masking=True):

    '''
    plot correlation matrix between gw and CNN-learned model mismatch 
    '''
    print 'max gw-grace correlation', np.nanmax(gwcorrmat)
    print 'min gw-grace correlation', np.nanmin(gwcorrmat)
    
    cmap = ListedColormap((sns.diverging_palette(240, 10, s=80, l=55, n=9)).as_hex())

    fig,ax = plt.subplots(figsize=(10,6), dpi=300)
    pp = np.zeros(gldas.mask.shape)+np.NaN
    pp[gldas.mask==1] = 1.0
    if masking:        
        temp = np.multiply(gwcorrmat, pp)
        im=ax.imshow(temp, cmap, origin='lower', vmax=0.8, vmin=-0.8)
        #plot CDF
        corrvec=gwcorrmat[gldas.gwvalidcells]
        #remove nan cells
        corrvec = corrvec[np.where(np.isnan(corrvec)==False)[0]]
        cdf = ECDF(corrvec)        
        #[left, bottom, width, height]
        figpos = [0.52, 0.7, 0.18, 0.16]
        #sns.set_style("whitegrid")   
        sns.set_style("ticks", {"xtick.major.size": 6, "ytick.major.size": 6})     
        ax2 = fig.add_axes(figpos)    
        ax2.set_ylim(0,1)
        ax2.grid(True)    
        ax2.plot(cdf.x, cdf.y, linewidth=1.5)
        ax2.set_xlabel('Correlation')
        ax2.set_ylabel('CDF')      
    else:
        im=ax.imshow(gwcorrmat, cmap, origin='lower')

    cx,cy = gldas.getBasinBoundForPlotting()
    ax.plot(cx,cy, '-', color='#7B7D7D')
    plt.colorbar(im, orientation="horizontal", fraction=0.046, pad=0.1, ax=ax)

    plt.savefig('gwcorr{0}.png'.format(label),dpi=fig.dpi, transparent=True, frameon=False)

def doTesting(model, label, gldas, X_train, X_test, Y_test, Xa_train=None, Xa_test=None, 
              Xnd_train=None, Xnd_test=None, n_p=3, nTrain=106, pixel=False):
    
    if Xa_train is None:
        #using only gldas data
        ypred = model.predict(X_test, batch_size=batch_size, verbose=0)
        ytrain = model.predict(X_train, batch_size=batch_size, verbose=0)
    elif Xnd_train is None:
        #using gldas and precip data
        ypred = model.predict([Xa_test, X_test], batch_size=batch_size, verbose=0)
        ytrain = model.predict([Xa_train, X_train], batch_size=batch_size, verbose=0)
    else:
        #using gldas, precip, and ndvi data
        ypred = model.predict([Xa_test, Xnd_test, X_test], batch_size=batch_size, verbose=0)
        ytrain = model.predict([Xa_train, Xnd_train, X_train], batch_size=batch_size, verbose=0)
    #calculate correlation matrix 
    #set validationOnly to True to calculate only for validation Period
    #Be careful, need to turn this off when plotting comparison with NOAH
    #corrmat = calcRMat(gldas, ytrain, ypred, X_train, X_test, n_p, validationOnly=True)
    corrmat = calcRMat(gldas, ytrain, ypred, X_train, X_test, n_p, validationOnly=False)
    
    #prepare gw data

    india = ProcessIndiaDB(reLoad=False)
    gldas.getGridAverages(india, reLoad=False)
    #do gw-grace correlation matrix
    gwcorrmat = calcCNN_GW_RMat(gldas, ytrain, ypred, n_p)
   
    print ypred.shape
    mse = np.zeros((Y_test.shape[0]))
    for i in range(ypred.shape[0]):        
        mse[i]=RMSE(Y_test[i,:,:], ypred[i,:,:])
    #print 'all rmse', mse  
    print 'RMSE=%s' % np.mean(mse)

    twsTrain = calculateBasinAverage(gldas, ytrain)    
    twsPred = calculateBasinAverage(gldas, ypred)    
    
    #calculate correlation
    predAll = np.r_[twsTrain, twsPred]
        
    #perform correlation up to 2016/12 [0, 154), total = 177
    
    print '------------------------------------------------------------------------------------'
    nEnd = 177-n_p
    
    print 'All: predicted vs. grace_raw', pearsonr(gldas.twsnldas[n_p-1:nEnd]-predAll[:nEnd-n_p+1], gldas.twsgrace[n_p-1:nEnd])
    print 'All: gldas vs. grace_raw', pearsonr(gldas.twsnldas[n_p-1:nEnd], gldas.twsgrace[n_p-1:nEnd])
    print 'Training: predicted vs. grace_raw', pearsonr(gldas.twsnldas[n_p-1:nTrain]-predAll[:nTrain-n_p+1], gldas.twsgrace[n_p-1:nTrain])
    print 'Training: gldas vs. grace_raw', pearsonr(gldas.twsnldas[n_p-1:nTrain], gldas.twsgrace[n_p-1:nTrain])
    print 'Testing: predicted vs. grace_raw', pearsonr(gldas.twsnldas[nTrain:nEnd]-predAll[nTrain-n_p+1:nEnd-n_p+1], gldas.twsgrace[nTrain:nEnd]) 
    print 'Testing: gldas vs. grace_raw', pearsonr(gldas.twsnldas[nTrain:nEnd], gldas.twsgrace[nTrain:nEnd])
    print '------------------------------------------------------------------------------------'

    sns.set_style("white")
    fig=plt.figure(figsize=(6,3), dpi=250)
    ax = plt.subplot(111)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    T0 = 2002+4.0/12
    rngTrain = T0+np.array(range(n_p-1,nTrain))/12.0
    rngTest = T0+np.array(range(nTrain, nEnd))/12.0
    rngFull = T0+np.array(range(n_p-1,nEnd))/12.0
    ax.plot(rngTrain, gldas.twsnldas[n_p-1:nTrain]-predAll[:nTrain-n_p+1], '--', color='#2980B9', label='CNN Train')
    ax.plot(rngTrain, gldas.twsgrace[n_p-1:nTrain], '-', color='#7D3C98', label='GRACE Train')
    ax.plot(rngTest, gldas.twsnldas[nTrain:nEnd]-predAll[nTrain-n_p+1:nEnd-n_p+1], '--o', color='#2980B9', label='CNN Test',markersize=3.5)
    ax.plot(rngTest, gldas.twsgrace[nTrain:nEnd], '-o', color='#7D3C98', label='GRACE Test', markersize=3.5)
    ax.plot(rngFull, gldas.twsnldas[n_p-1:nEnd], ':', color='#626567', label='NOAH')
    ax.axvspan(xmin=T0+(nTrain-1.5)/12, xmax=T0+(nTrain+0.5)/12, facecolor='#7B7D7D', linewidth=2.0)
    ax.grid(False)
    ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.65), borderaxespad=0., fancybox=True, frameon=True)
    labeldict={'simple':'(a)', 'vgg16':'(b)', 'comp':'(c)', 'ndvicnn':'(d)'}
    ax.text(0.02, 0.90, labeldict[label], fontsize=10, transform=ax.transAxes)
    plt.savefig('gldas_timeseriesplot%s.png' % label, dpi=fig.dpi)

    #save results for plotting
    pkl.dump([nTrain,rngTrain,rngTest, rngFull, gldas.twsnldas[n_p-1:nTrain]-predAll[:nTrain-n_p+1],
              gldas.twsgrace[n_p-1:nTrain],
              gldas.twsnldas[nTrain:nEnd]-predAll[nTrain-n_p+1:nEnd-n_p+1],
              gldas.twsgrace[nTrain:nEnd],
              gldas.twsnldas[n_p-1:nEnd]], open('basincorr{0}.pkl'.format(label), 'wb'))

    #plotOutput(nldas, ypred, Y_test, X_test)
    #plot correlation matrix
    plotCorrmat(label, gldas, corrmat)
    #
    plotGwCorrmat(label, gldas, gwcorrmat, True)
    #
    '''
    if pixel:
        #plot time series from selected pixel
        india = ProcessIndiaDB(reLoad=False)
        grace = GRACEData(reLoad=False, watershed='india')
        #loc=[84, 24]
        loc = [78, 32]
        cx, cy, gcx, gcy, ts, graceTS  = gldas.plotWells(india.dfwellInfo, gl=grace, loc=loc)        

        t1 = calculateBasinPixel(gldas, ytrain, cy, cx)
        t2 = calculateBasinPixel(gldas, ypred, cy, cx)
        #tsAll start with np-1, so pad with nan
        tsAll = np.r_[np.zeros((n_p-1))+np.NaN, t1, t2]
        #corrected = tsAll[n_p-1:nEnd]+ts[n_p-1:nEnd]
        corrected = ts[:nEnd] - tsAll[:nEnd]
        print len(corrected)
        nMonths = len(corrected)
        rng = pd.date_range('4/1/2002', periods=nMonths, freq='M')
        predDF = pd.DataFrame({'twsCorrected':corrected, 'noah':ts[:nEnd], 'grace':graceTS[:nEnd]}, index=rng)        
        
        ax = predDF.plot()
        ax.grid(False)
        #convert to dateframe
        gwDF = india.getAvg(loc[1],loc[0], 0.5/2.0)
        if not gwDF is None:
            ax3=ax.twinx()
            gwDF.plot(ax=ax3, style='ro:')
            ax3.grid(False)

        plt.savefig('pixeltest.png')
    '''
def main(retrain=False, modelnum=1, modeloption=None):
    watershed='indiabig'
    watershedActual = 'indiabang'
    n_p = 3 
    nTrain = 125
    #comments: cnndriver obtains the best results @n=80
    if modelnum==1:
        LSTMDriver(watershedName=watershed, watershedInner=watershedActual, retrain=retrain)  
    elif modelnum==2:
        CNNDriver(watershedName=watershed, watershedInner=watershedActual, n_p=n_p, nTrain=nTrain, retrain=retrain, modelOption=modeloption)
    elif modelnum==3:
        CNNCompositeDriver(watershedName=watershed, watershedInner=watershedActual, n_p=n_p, nTrain=nTrain, retrain=retrain)
    elif modelnum==4:
        UnetDriver(watershedName=watershed, watershedInner=watershedActual, retrain=retrain)
    elif modelnum==5:
        CNNNDVICompositeDriver(watershedName=watershed, watershedInner=watershedActual, retrain=retrain, modelOption=modeloption)
        
if __name__ == "__main__":
    main(retrain=False, modelnum=2, modeloption=1)#vgg16
    main(retrain=False, modelnum=2, modeloption=2)#simple
    main(retrain=False, modelnum=3)#composite, P
    main(retrain=False, modelnum=5, modeloption=1)#composite, NDVI