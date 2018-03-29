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
K.set_session(sess)
from keras.utils import plot_model

import seaborn as sns
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from nldasColorado import GRACEData, NLDASData


'''
dim_ordering issue:
- 'th'-style dim_ordering: [batch, channels, depth, height, width]
- 'tf'-style dim_ordering: [batch, depth, height, width, channels]
'''
nb_epoch = 15    # number of epoch at training (cont) stage
batch_size = 15  # batch size

# C is number of channel
# Nf is number of antecedent frames used for training
C=1
N=64
MASKVAL=np.NaN
#for reproducibility
from numpy.random import seed
seed(1111)

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
        plot_model(model, to_file='cnn1model.png')

    return model

def seqCNN3(seq_len=3, summary=False,backend='tf'):
    if backend == 'tf':
        input_shape=(N, N, seq_len) # l, h, w, c
    else:
        input_shape=(seq_len, N, N) # c, l, h, w
        
    model = Sequential()
    model.add(Conv2D(128, kernel_size=(3, 3), input_shape=input_shape, padding='same',name='conv1',activation='relu'))
    model.add(Conv2D(64, kernel_size=(3, 3), padding='same',activation='relu'))
    model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu'))
    
    model.add(Conv2D(1, kernel_size=(1, 1), padding='same'))
    model.add(Activation('tanh'))
    model.add(Flatten())
    model.add(Reshape((N,N)))

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


def CNNDriver(watershedName, retrain=False):
    grace = GRACEData(watershed=watershedName)    
    nldas = NLDASData(watershed=watershedName)
    nldas.loadStudyData(reloadData=False)    

    n_p = 3
    model = seqCNN3(seq_len=n_p, summary=True)
    X_train,Y_train,X_test,Y_test,Xval = nldas.formMatrix2D(gl=grace, n_p=n_p, masking=True, nTrain=106)
    
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
        model.save_weights('{0}model_weights.h5'.format(watershedName))
    else:
        model.load_weights('{0}model_weights.h5'.format(watershedName))
    
    mse = np.zeros((Y_test.shape[0]))
    ypred = model.predict(X_test, batch_size=batch_size, verbose=0)
    print ypred.shape
    for i in range(ypred.shape[0]):        
        mse[i]=RMSE(Y_test[i,:,:], ypred[i,:,:])
        
    print 'RMSE=%s' % np.mean(mse)
    #comment out the following to calculate basin average time series

    ytrain = model.predict(X_train, batch_size=batch_size, verbose=0)
    twsTrain = calculateBasinAverage(nldas, ytrain)    
    for item in twsTrain:
        print item
    
    twsPred = calculateBasinAverage(nldas, ypred)    
    twsIn = calculateBasinAverage(nldas, Xval)
    twsnldasavg = nldas.twsnldas[-Y_test.shape[0]:]   
    twsgrace = nldas.twsgrace[-Y_test.shape[0]:]

    for i in range(len(twsPred)):
        print '%s' % (twsPred[i])
        
    #plotOutput(nldas, ypred, Y_test, X_test)

def main(retrain=False, modelnum=1):
    watershed='colorado'
    if modelnum==1:
        CNNDriver(watershedName=watershed, retrain=retrain)
        
if __name__ == "__main__":
    main(retrain=True, modelnum=1)