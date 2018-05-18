#Author: Alex Sun
#Date: 1/24/2018
#Purpose: train the gldas-grace model
#date: 2/16/2018 adapt for the gldas
#date: 3/7/2018 adapt for global
#date: 3/8/2018 adapt for local bigger area study
#corrected the Dropout layer problem switch ndvivgg16 to ndvivgg16SRT version
#=======================================================================
import numpy as np
import tensorflow as tf
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
from gldasBasin import GRACEData, GLDASData, NDVIData
import gldasBasin

from keras.callbacks import ReduceLROnPlateau,EarlyStopping
from keras.utils import plot_model

from scipy.stats.stats import pearsonr
import sklearn.linear_model as skpl
import pandas as pd
from ProcessIndiaDB import ProcessIndiaDB
from numpy.random import seed

import pickle as pkl
from statsmodels.distributions.empirical_distribution import ECDF
from unet import Unet
from matplotlib.ticker import (FixedLocator, FormatStrFormatter,
                               AutoMinorLocator)
'''
dim_ordering issue:
- 'th'-style dim_ordering: [batch, channels, depth, height, width]
- 'tf'-style dim_ordering: [batch, depth, height, width, channels]
'''
nb_epoch = 60   # number of epoch at training (cont) stage
batch_size = 15  # batch size

# C is number of channel
# Nf is number of antecedent frames used for training
C=1
N=gldasBasin.N
MASKVAL=np.NaN
reduce_lr = ReduceLROnPlateau(monitor='rmse', factor=0.5,
              patience=2, min_lr=0.000001)

early_stop = EarlyStopping(monitor='rmse',
                           patience=3,
                           min_delta=0, 
                           verbose=1,
                           mode='auto')




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
    x = Reshape((N,N))(x)
    model = Model(input=input, output=x)

    if summary:
        print(model.summary())
        plot_model(model, to_file='vgg16model1.png', show_shapes=True)
    return model

def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))    

def MSE(y_test, y_pred):    
    return np.mean(np.square(y_test - y_pred))
    
def NSE(y_test, y_pred):
    '''
    @param y_test, observation for validation
    @param y_pred, predicted values 
    '''
    term1 = np.sum(np.square(y_test - y_pred))
    term2 = np.sum(np.square(y_test - np.mean(y_test)))
    return 1.0 - term1/term2

def NSE_M(y_test, y_pred):
    '''
    modified NSE
    @param y_test, observation for validation
    @param y_pred, predicted values 
    '''
    term1 = np.sum(np.abs(y_test - y_pred))
    term2 = np.sum(np.abs(y_test - np.mean(y_test)))
    return 1.0 - term1/term2

def r2(y_test, y_pred):
    #calculates coefficient of determination 
    SS_res =  K.sum(K.square( y_test-y_pred ))
    SS_tot = K.sum(K.square( y_test - K.mean(y_test) ) )
    return ( 1.0 - SS_res/(SS_tot + K.epsilon()) )

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

def modelTrain(model, retrain, watershedName, label, Xin, Yin, weightfile):
    print (model.summary())
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
                        
        #model.compile(loss='mse', optimizer=solver, metrics=[rmse, r2])
        model.compile(loss=moo, optimizer=solver, metrics=[rmse, r2])
        history = model.fit(Xin, Yin,
                            epochs=nb_epoch,
                            batch_size=batch_size,
                            shuffle=True,
                            verbose=1, callbacks=[reduce_lr])
        model.save_weights(weightfile)
        np.save('{0}history.npy'.format(label), history.history)
    else:
        model.load_weights(weightfile)

def moo(y_true, y_pred):
    #calculates coefficient of determination 
    #SS_res =  K.sum(K.square( y_true-y_pred ))
    #SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    SS_res =  K.sum(K.abs( y_true-y_pred ))
    SS_tot = K.sum(K.abs( y_true - K.mean(y_true) ) )

    #need to minimize this toward zero
    nse = SS_res/(SS_tot + K.epsilon())
    mse = K.mean(K.abs(y_pred - y_true), axis=-1)
    return nse

def CNNCompositeDriver(watershedName, watershedInner, n_p=3, nTrain=125, modelOption=1, 
                       retrain=False, realizationNum=None):
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
    
    if modelOption==2:
        #note vgg16 only allows 3 channels
        inLayer = Input(shape=input_shape, name='input')
        outLayer = Conv2D(2, kernel_size=(3, 3), padding='same', activation='relu')(inLayer)
            
        inputPLayer = Input(shape=input_shape, name='inputP')
        outPLayer = Conv2D(1, kernel_size=(3, 3), padding='same',activation='relu')(inputPLayer)
        x = keras.layers.concatenate([outPLayer, outLayer], axis=-1)  
        x = CNNvgg16(x, input_shape = [N, N, inLayer._keras_shape[3]])
        label = 'compvggSRT'
        
    model = Model(inputs=[inputPLayer, inLayer], outputs=[x])
    weightfile = 'glo{0}{1}model_weights{2}.h5'.format(watershedName, label, realizationNum)
    modelTrain(model, retrain, watershedName, label, 
               Xin=[Xp_train, X_train], Yin=Y_train, weightfile=weightfile)

    doTesting(model, label, nldas, X_train, X_test, Y_train, Y_test, Xp_train, Xp_test, n_p=n_p, 
              nTrain=nTrain, nrz=realizationNum)
    
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

def CNNvgg16(inputLayer, input_shape=None):
    '''
    the transfer learning model for use with composite model
    '''
    if input_shape is None:
        input_shape=[N, N, inputLayer._keras_shape[3]] # h, w, c
    print ('input shape = ', input_shape)
    print (inputLayer._keras_shape)

    model_vgg16_conv = VGG16(include_top=False, input_shape=input_shape, weights='imagenet')
    for layer in model_vgg16_conv.layers:
        layer.trainable = False
        
    output_vgg16_conv = model_vgg16_conv(inputLayer)
    #Add the fully-connected layers 
    x = Flatten(name='flatten')(output_vgg16_conv)
    x = Dropout(0.2)(x)    
    x = Dense(N*N, activation='tanh', name='fc1')(x)
    x = Reshape((N,N))(x)
    
    return x


def CNNNDVICompositeDriver(watershedName, watershedInner, n_p=3, nTrain=125, 
                           modelOption=1, retrain=False, realizationNum=''):
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
        
        label = 'ndvivggSRT'
    
                                                                                                        
    model = Model(inputs=[inputPLayer, inputNDVILayer, inLayer], outputs=[x])
    weightfile = 'glo{0}ndvimodel_weights{1}{2}.h5'.format(watershedName,label,realizationNum)
    modelTrain(model, retrain, watershedName, label, Xin=[Xp_train, Xnd_train, X_train], 
               Yin=Y_train, weightfile=weightfile)

        

    doTesting(model, label, nldas, X_train, X_test, Y_train, Y_test, Xp_train, Xp_test,
              Xnd_train, Xnd_test, 
              n_p=n_p, nTrain=nTrain, pixel=False, nrz=realizationNum)
    
def CNNDriver(watershedName, watershedInner, n_p=3, nTrain=125, retrain=False, modelOption=1, realizationNum=None):
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
        
    X_train,Y_train,X_test,Y_test,_ = nldas.formMatrix2D(gl=grace, n_p=n_p, 
                                                            masking=isMasking, nTrain=nTrain)
    
    weightfile='glo{0}model_weights{1}{2}.h5'.format(watershedName, ioption, realizationNum)
    modelTrain(model, retrain, watershedName, label, Xin=X_train, 
               Yin=Y_train, weightfile=weightfile)
         
    doTesting(model, label, nldas, X_train, X_test, Y_train, Y_test, n_p=n_p, nTrain=nTrain, 
              pixel=False, nrz=realizationNum)
         
  
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
        print ('array dimensions', mat.shape[0], gldas.graceArr.shape[0])
        for i in range(mat.shape[1]):
            corrmat[i],_ = pearsonr(mat[:,i], gldas.graceArr[n_p-1:nMonths+n_p,i])
    else:
        #correlation over the validation period only
        matPred = getMat(Ypred, Xpred)
        #concatenate along the rows
        mat = matPred
        corrmat = np.zeros((gldas.nvalidCells), dtype='float64')
        nTrain = Ytrain.shape[0]
        print ('array dimensions', mat.shape[0], gldas.graceArr.shape[0])
        for i in range(mat.shape[1]):
            corrmat[i],_ = pearsonr(mat[:,i], gldas.graceArr[nTrain:-1,i])
        
    #convert to image
    temp = np.zeros((N,N))+np.NaN
    temp[gldas.validCells] = corrmat

    return temp


def calcNSEMat(gldas,  Ytrain, Ypred, Xtrain, Xpred, n_p, validationOnly=False):
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
        nsemat = np.zeros((gldas.nvalidCells), dtype='float64')
        nMonths = mat.shape[0]
        print ('array dimensions', mat.shape[0], gldas.graceArr.shape[0])
        for i in range(mat.shape[1]):
            nsemat[i] = NSE(y_pred=mat[:,i], y_test=gldas.graceArr[n_p-1:nMonths+n_p,i])
    else:
        #correlation over the validation period only
        matPred = getMat(Ypred, Xpred)
        #concatenate along the rows
        mat = matPred
        nsemat = np.zeros((gldas.nvalidCells), dtype='float64')
        nTrain = Ytrain.shape[0]
        print ('array dimensions', mat.shape[0], gldas.graceArr.shape[0])
        for i in range(mat.shape[1]):
            nsemat[i] = NSE(y_pred=mat[:,i], y_test=gldas.graceArr[nTrain:-1,i])
        
    #convert to image
    temp = np.zeros((N,N))+np.NaN
    temp[gldas.validCells] = nsemat

    return temp


def plotCorrmat(label, gldas, corrected=None, masking=True):
    '''
    plot cnn vs. grace correlation grid map 
    '''
    #plot image
    cmap = ListedColormap((sns.diverging_palette(240, 10, n=9)).as_hex())
    
    fig, axes =plt.subplots(1, 3, figsize=(12,6), dpi=300) 

    pp = np.zeros(gldas.mask.shape)+np.NaN
    pp[gldas.mask==1] = 1.0
    ppActual = np.zeros(gldas.innermask.shape)+np.NaN
    #masking for india itself
    ppActual[gldas.innermask==1]=1.0

    if masking:
        vmin=-0.8; vmax=1.0        
        temp = np.multiply(gldas.gldas_grace_R, pp)
        #im=plt.imshow(temp, cmap, origin='lower', alpha=0.9, interpolation='bilinear', vmin=vmin,vmax=vmax)
        temp = np.multiply(gldas.gldas_grace_R, ppActual)
        im=axes[0].imshow(temp, cmap, origin='lower', alpha=0.95, interpolation='bilinear',vmin=vmin,vmax=vmax)      
    else:
        im=axes[0].imshow(gldas.gldas_grace_R, cmap, origin='lower')
        
    cx,cy = gldas.getBasinBoundForPlotting()

    axes[0].plot(cx,cy, '-', color='#7B7D7D')
    plt.colorbar(im, orientation="horizontal", fraction=0.046, pad=0.1, ax=axes[0])
    axes[0].set_title(r'(a) $\rho_{NOAH/GRACE}$')
    
    if not corrected is None:
        if masking:
            vmin=-0.4; vmax=1.0
            corrected = np.multiply(corrected, pp)
            correctedInner = np.multiply(corrected, ppActual)            
            gldas_grace_R_inner=np.multiply(gldas.gldas_grace_R, ppActual)
            #save for CDF plot
            np.save('corrdist{0}.npy'.format(label), [correctedInner[ppActual==1],gldas_grace_R_inner[ppActual==1]])
        #compute correlation between GRACE and learned values
        #im=plt.imshow(corrected, cmap, origin='lower', alpha=0.9, interpolation='bilinear',vmin=vmin,vmax=vmax)
        im=axes[1].imshow(correctedInner, cmap, origin='lower', alpha=0.95, interpolation='bilinear',vmin=vmin,vmax=vmax)
        plt.colorbar(im, orientation="horizontal", fraction=0.046, pad=0.1, ax=axes[1])
        axes[1].plot(cx,cy, '-', color='#7B7D7D')
        axes[1].set_title(r'(b) $\rho_{VGG16/GRACE}$')
        
        vmin=-0.6;vmax=0.6
        dd = corrected-gldas.gldas_grace_R
        #im=plt.imshow(dd, cmap, origin='lower',alpha=0.9, interpolation='bilinear',vmin=vmin,vmax=vmax)
        dd = np.multiply(dd, ppActual)
        im=axes[2].imshow(dd, cmap, origin='lower',alpha=0.95, interpolation='bilinear',vmin=vmin,vmax=vmax)    
        plt.colorbar(im, orientation="horizontal", fraction=0.046, pad=0.1, ax=axes[2])
        axes[2].plot(cx,cy, '-', color='#7B7D7D')
        axes[2].set_title(r'(c) $\Delta\rho$')
        print (np.nanmin(dd))
        print (np.nanmax(dd))

    for i in range(3):
        axes[i].set_xlim([20, 120])
        axes[i].set_ylim([0, 100])

    plt.tight_layout(h_pad=1.02)
    plt.savefig('cnn_grace_corr%s.png' % label, dpi=fig.dpi,transparent=True, frameon=False)

def plotNSEmat(label, gldas, corrected=None, masking=True):
    '''
    plot cnn vs. grace NSE grid map 
    '''
    #plot image
    cmap = ListedColormap((sns.diverging_palette(240, 10, n=9)).as_hex())
    
    fig, axes =plt.subplots(1, 3, figsize=(12,6), dpi=300) 

    pp = np.zeros(gldas.mask.shape)+np.NaN
    pp[gldas.mask==1] = 1.0
    ppActual = np.zeros(gldas.innermask.shape)+np.NaN
    #masking for india itself
    ppActual[gldas.innermask==1]=1.0

    if masking:
        vmin=-1.0; vmax=1.0        
        temp = np.multiply(gldas.gldas_grace_NSE, pp)
        temp = np.multiply(gldas.gldas_grace_NSE, ppActual)
        im=axes[0].imshow(temp, cmap, origin='lower', alpha=0.95, interpolation='bilinear',vmin=vmin,vmax=vmax)      
    else:
        im=axes[0].imshow(gldas.gldas_grace_NSE, cmap, origin='lower')
        
    cx,cy = gldas.getBasinBoundForPlotting()

    axes[0].plot(cx,cy, '-', color='#7B7D7D')
    plt.colorbar(im, orientation="horizontal", fraction=0.046, pad=0.1, ax=axes[0])
    axes[0].set_title(r'(d) $NSE_{NOAH/GRACE}$')
    
    if not corrected is None:
        if masking:
            vmin=-0.8; vmax=1.0
            corrected = np.multiply(corrected, pp)
            correctedInner = np.multiply(corrected, ppActual)            
            gldas_grace_NSE=np.multiply(gldas.gldas_grace_NSE, ppActual)
            #save for CDF plot
            np.save('nsemat{0}.npy'.format(label), [correctedInner[ppActual==1],gldas_grace_NSE[ppActual==1]])
        #compute correlation between GRACE and learned values
        #im=plt.imshow(corrected, cmap, origin='lower', alpha=0.9, interpolation='bilinear',vmin=vmin,vmax=vmax)
        im=axes[1].imshow(correctedInner, cmap, origin='lower', alpha=0.95, interpolation='bilinear',vmin=vmin,vmax=vmax)
        plt.colorbar(im, orientation="horizontal", fraction=0.046, pad=0.1, ax=axes[1])
        axes[1].plot(cx,cy, '-', color='#7B7D7D')
        axes[1].set_title(r'(e) $NSE_{VGG16/GRACE}$')
        
        vmin=-0.6;vmax=1.0
        dd = corrected-gldas.gldas_grace_NSE
        #im=plt.imshow(dd, cmap, origin='lower',alpha=0.9, interpolation='bilinear',vmin=vmin,vmax=vmax)
        dd = np.multiply(dd, ppActual)
        im=axes[2].imshow(dd, cmap, origin='lower',alpha=0.95, interpolation='bilinear',vmin=vmin,vmax=vmax)    
        plt.colorbar(im, orientation="horizontal", fraction=0.046, pad=0.1, ax=axes[2])
        axes[2].plot(cx,cy, '-', color='#7B7D7D')
        axes[2].set_title(r'(f) $\Delta NSE$')
        print (np.nanmin(dd))
        print (np.nanmax(dd))

    for i in range(3):
        axes[i].set_xlim([20, 120])
        axes[i].set_ylim([0, 100])

    plt.tight_layout(h_pad=1.02)
    plt.savefig('cnn_grace_nse%s.png' % label, dpi=fig.dpi,transparent=True, frameon=False)
        
def calculateBasinAverage(nldas, Y, subbasin=None):
    '''
    Y is either predicted or test tensor
    '''
    if subbasin is None:
        nvalidCell = nldas.nActualCells
        mask = nldas.innermask
    else:
        if subbasin == 'indus':
            nvalidCell = len(np.where(nldas.indusmask==1)[0]) 
            mask = nldas.indusmask            
        elif subbasin== 'ganges':
            nvalidCell = len(np.where(nldas.gangesmask==1)[0])
            mask = nldas.gangesmask
    
    tws_avg = np.zeros((Y.shape[0]))
    for i in range(Y.shape[0]):
        obj = backTransform(nldas, np.multiply(Y[i,:,:], mask))
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

def calcCNN_GW_RMatWellls(gldas, Ytrain, Ypred, n_p, gwdb):
    '''
    as05012018
    calculate correlation between each well's gwsa and the cn prediction 
    @param gldas, instance of gldas
    @param Ytrain, training target
    @param Ypred,  predicted target
    '''
    def getMat(Yin):
        mat = np.zeros((Yin.shape[0], N*N), dtype=np.float64)
        for i in range(Yin.shape[0]):
            obj = -backTransform(gldas, Yin[i,:,:])
            mat[i,:] = np.reshape(obj, (1,N*N))
        return mat
            
    matTrain = getMat(Ytrain)
    matPred = getMat(Ypred)
    #concatenate along the rows
    mat = np.r_[matTrain,matPred[1:]]
    #extract records corresponding to gw measurements
    #gldas mat time axis is from 2002/04, gw starts from 2005/01
    rng = np.array([33,37,40,43], dtype='int')
    b = np.array([], dtype='int')
    for i in range(2005,2014):
        b = np.r_[b, rng]
        rng = rng+12
    print ('corr indices', b)
    #need to shift to account for n_p
    #mat = mat[b-(n_p-1),:]
    mat = mat[b-(n_p-1),:]

    print ('in gwwell', mat.shape)
    gwdb = gldas.getGWDB(gwdb, reLoad=False)
    #now calculate correlation
    nwells = len(gwdb.dfwellInfo)    
    corrmat = np.zeros((nwells, 3))
    counter=0
    for item in gwdb.dfwellInfo['wellid'].values:
        #get water lvl
        df = gwdb.dfAll.loc[gwdb.dfAll['wellid']==item]
        #df = pd.DataFrame({'waterlvl':df['waterlvl'].values}, index= gwdb.rng)
        #df = df.interpolate(method='linear')
        #do correlation 
        tp = gwdb.dfwellInfo.loc[gwdb.dfwellInfo['wellid']==item, ['ix', 'iy']]
        ix = int(tp['ix'])
        iy = int(tp['iy'])
        #get grace data
        #gl = gracemat[:, ]  
        corrmat[counter, :] = [iy, ix, pearsonr(df['waterlvl'].values, mat[:, (iy-1)*N+ix])[0]]
        counter+=1
    
    return corrmat

def plotGwCorrmatWell(label, gldas, gwcorrmat):
    '''
    plot correlation matrix between individual wells and CNN result  
    '''
    print ('max well-grace correlation', np.nanmax(gwcorrmat[:,2]))
    print ('min well-grace correlation', np.nanmin(gwcorrmat[:,2]))
    
    cmap = ListedColormap((sns.diverging_palette(240, 10, s=80, l=55, n=9)).as_hex())

    fig,ax = plt.subplots(figsize=(10,10), dpi=300)
    #define markersize
    ms = np.zeros((gwcorrmat.shape[0]))+15
    sc = ax.scatter(x=gwcorrmat[:,1], y = gwcorrmat[:,0], c=gwcorrmat[:,2], cmap=cmap, s=ms)
    #plot CDF
    corrvec=gwcorrmat[:,2]
    #remove nan cells
    corrvec = corrvec[np.where(np.isnan(corrvec)==False)[0]]
    cdf = ECDF(corrvec)        
    #[left, bottom, width, height]
    figpos = [0.54, 0.73, 0.26, 0.12]
    #sns.set_style("whitegrid")   
    sns.set_style("ticks", {"xtick.major.size": 6, "ytick.major.size": 6})
    ax.set_xlim([20, 120])
    ax.set_ylim([0, 100])
    ax.set(adjustable='box-forced', aspect='equal')
    
    ax2 = fig.add_axes(figpos)    
    ax2.plot(cdf.x, cdf.y, linewidth=1.5)
    ax2.set_xlabel('Corr')
    ax2.set_ylabel('CDF')      
    
    ax2.set_ylim(0,1)
    ax2.set_xlim(-1,1)               
    ax2.grid(True, which='both')    
    
    cx,cy = gldas.getBasinBoundForPlotting()
    ax.plot(cx,cy, '-', color='#7B7D7D')
    plt.colorbar(sc, orientation="horizontal", fraction=0.046, pad=0.1, ax=ax)
    
    plt.savefig('gwcorrwell{0}.eps'.format(label),dpi=fig.dpi, transparent=True, frameon=False)

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
    print ('corr indices', b)
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
    print ('mean gw corr is', np.nanmean(corrmat))
    return temp

def plotGwCorrmat(label, gldas, gwcorrmat, masking=True):

    '''
    plot correlation matrix between gw and CNN-learned model mismatch 
    '''
    print ('max gw-grace correlation', np.nanmax(gwcorrmat))
    print ('min gw-grace correlation', np.nanmin(gwcorrmat))
    
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

def getUnetModel(n_p=3, summary=False, inputLayer=None, level=7):
    inputs = inputLayer
    if level==7:
        conv1 = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(inputs)
        conv1 = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
        conv2 = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(pool1)
        conv2 = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
        conv3 = Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same')(pool2)
        conv3 = Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
        conv4 = Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same')(pool3)
        conv4 = Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same')(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    
        conv5 = Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same')(pool4)
        conv5 = Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same')(conv5)
    
        up6 = keras.layers.concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=-1)
        conv6 = Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same')(up6)
        conv6 = Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same')(conv6)
        up7 = keras.layers.concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=-1)
        conv7 = Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same')(up7)
        conv7 = Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same')(conv7)
    
        up8 = keras.layers.concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=-1)
        conv8 = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(up8)
        conv8 = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(conv8)
    
        up9 = keras.layers.concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=-1)
        conv9 = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(up9)
        conv9 = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(conv9)
    elif level==8:
        conv1 = Conv2D(16, (3, 3), padding="same", activation="relu")(inputs)
        conv1 = Conv2D(16, (3, 3), padding="same", activation="relu")(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(32, (3, 3), padding="same", activation="relu")(pool1)
        conv2 = Conv2D(32, (3, 3), padding="same", activation="relu")(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(64, (3, 3), padding="same", activation="relu")(pool2)
        conv3 = Conv2D(64, (3, 3), padding="same", activation="relu")(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(128, (3, 3), padding="same", activation="relu")(pool3)
        conv4 = Conv2D(128, (3, 3), padding="same", activation="relu")(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = Conv2D(256, (3, 3), padding="same", activation="relu")(pool4)
        conv5 = Conv2D(256, (3, 3), padding="same", activation="relu")(conv5)

        up6 = keras.layers.concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=3) # concat_axis=3 for Tensorflow vs 1 for theano
        conv6 = Conv2D(128, (3, 3), padding="same", activation="relu")(up6)
        conv6 = Conv2D(128, (3, 3), padding="same", activation="relu")(conv6)

        up7 = keras.layers.concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=3)
        conv7 = Conv2D(64, (3, 3), padding="same", activation="relu")(up7)
        conv7 = Conv2D(64, (3, 3), padding="same", activation="relu")(conv7)

        up8 = keras.layers.concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=3)
        conv8 = Conv2D(32, (3, 3), padding="same", activation="relu")(up8)
        conv8 = Conv2D(32, (3, 3), padding="same", activation="relu")(conv8)

        up9 = keras.layers.concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=3)
        conv9 = Conv2D(16, (3, 3), padding="same", activation="relu")(up9)
        conv9 = Conv2D(16, (3, 3), padding="same", activation="relu")(conv9)
    else:
        raise Exception('invalid unet level')
    conv9 = Dropout(0.2)(conv9)
    conv10 = Conv2D(1, kernel_size=(1, 1), activation='tanh')(conv9)
    x = Flatten()(conv10)
    x = Reshape((N,N))(x)

    return x

def UnetDriver(watershedName, watershedInner, retrain=False, n_p=3, nTrain=125, 
               modeloption=1, nrz=None):
    
    isMasking=True
    grace = GRACEData(reLoad=False, watershed=watershedName)    
    nldas = GLDASData(watershed=watershedName,watershedInner=watershedInner)
    nldas.loadStudyData(reloadData=False)    
    ndvi = NDVIData(reLoad=False, watershed=watershedName)
    
    input_shape = (N,N,n_p)

    X_train,Y_train,X_test,Y_test,_ = nldas.formMatrix2D(gl=grace, n_p=n_p, masking=True, nTrain=nTrain)
    Xp_train, Xp_test = nldas.formPrecip2D(n_p=n_p, masking=isMasking, nTrain=nTrain)
    Xnd_train, Xnd_test = nldas.formNDVI2D(ndvi, n_p=n_p, masking=isMasking, nTrain=nTrain)
    
    level = 7
    if modeloption==1:        
        inputLayer = Input(shape=input_shape, name='input')
        x = getUnetModel(n_p=n_p, summary=True, inputLayer=inputLayer, level=level)
        model = Model(inputs=[inputLayer], outputs=[x])
        label = 'unet'    
        inputArr = X_train    
        
    elif modeloption==2:
        inLayer = Input(shape=input_shape, name='input')
        outLayer = Conv2D(16, kernel_size=(3, 3), padding='same', activation='relu')(inLayer)
            
        inputPLayer = Input(shape=input_shape, name='inputP')
        outPLayer = Conv2D(16, kernel_size=(3, 3), padding='same',activation='relu')(inputPLayer)
        x = keras.layers.concatenate([outPLayer, outLayer], axis=-1)  
        x = getUnetModel(n_p, summary=True, inputLayer=x, level=level)
        label = 'unetp'
        inputArr = [Xp_train, X_train]
        model = Model(inputs=[inputPLayer, inLayer], outputs=[x])
        
    elif modeloption==3:
        inLayer = Input(shape=input_shape, name='input')
        outLayer = Conv2D(16, kernel_size=(3, 3), padding='same', activation='relu')(inLayer)
            
        inputPLayer = Input(shape=input_shape, name='inputP')
        outPLayer = Conv2D(16, kernel_size=(3, 3), padding='same',activation='relu')(inputPLayer)

        inputNDVILayer = Input(shape=(N,N,n_p), name='inputNDVI')
        outNDVILayer = Conv2D(16, kernel_size=(3, 3), padding='same',activation='relu')(inputNDVILayer)

        x = keras.layers.concatenate([outPLayer, outNDVILayer, outLayer], axis=-1)  
        x = getUnetModel(n_p, summary=True, inputLayer=x, level=level)
        label = 'unetndviSRT'
        inputArr = [Xp_train, Xnd_train, X_train]
 
        model = Model(inputs=[inputPLayer, inputNDVILayer, inLayer], outputs=[x])

    weightfile='{0}{1}model_weightsunet{2}.h5'.format(watershedName,label, nrz)
    modelTrain(model, retrain, watershedName, label, Xin=inputArr, 
               Yin=Y_train, weightfile=weightfile)
    
    if modeloption==1:
        doTesting(model, label, nldas, X_train, X_test, Y_train, Y_test, n_p=n_p, nTrain=nTrain, 
                  pixel=False, nrz=nrz) 
    elif modeloption==2:
        doTesting(model, label, nldas, X_train, X_test, Y_train, Y_test, Xp_train, Xp_test, 
                  n_p=n_p, nTrain=nTrain, pixel=False, nrz=nrz)
    elif modeloption==3:
        doTesting(model, label, nldas, X_train, X_test, Y_train, Y_test, Xp_train, Xp_test,
              Xnd_train, Xnd_test, n_p=n_p, nTrain=nTrain, pixel=False, nrz=nrz)

def doTesting(model, label, gldas, X_train, X_test, Y_train, Y_test, Xa_train=None, Xa_test=None, 
              Xnd_train=None, Xnd_test=None, n_p=3, nTrain=106, pixel=False, nrz=None):

    
    def calcStat(ts1, ts2):        
        return 'rho={0}, NSE={1}, NSEM={2}'.format(pearsonr(ts1,ts2)[0], NSE(y_test=ts2, y_pred=ts1), NSE_M(y_test=ts2, y_pred=ts1))
    
    def calcSubbasinTWS(basinname):
        #if basinname is None, the whole India will be returned
        twsTrain = calculateBasinAverage(gldas, ytrain, basinname)    
        twsPred = calculateBasinAverage(gldas, ypred, basinname)        
        #calculate correlation
        predAll = np.r_[twsTrain, twsPred]
        return predAll
    
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
    nsemat = calcNSEMat(gldas, ytrain, ypred, X_train, X_test, n_p, validationOnly=False)
    #prepare gw data
    india = ProcessIndiaDB(reLoad=False)
    gldas.getGridAverages(india, reLoad=False)
    #Do gw-grace correlation matrix
    gwcorrmat = calcCNN_GW_RMat(gldas, ytrain, ypred, n_p)
    doWellcorrel=False
    if doWellcorrel:
        wellcorrmat = calcCNN_GW_RMatWellls(gldas, ytrain, ypred, n_p, india)
        
    print (ypred.shape)
    mse = np.zeros((Y_test.shape[0]))
    for i in range(ypred.shape[0]):        
        mse[i]=MSE(Y_test[i,gldas.actualvalidCells], ypred[i,gldas.actualvalidCells])  
    print ('RMSE=%s' % np.sqrt(np.mean(mse)))
      
    #calculate correlation
    predAll = calcSubbasinTWS(basinname=None)
    #asun 04252018, get correlation for indus and ganges
    indusAll = calcSubbasinTWS(basinname='indus')        
    gangesAll = calcSubbasinTWS(basinname='ganges')
    
    #perform correlation up to 2016/12 [0, 154), total = 177
    
    print ('------------------------------------------------------------------------------------')
    nEnd = 177-n_p
    
    print ('All: predicted vs. grace_raw', calcStat(gldas.twsnldas[n_p-1:nEnd]-predAll[:nEnd-n_p+1], gldas.twsgrace[n_p-1:nEnd]))    
    print ('All: gldas vs. grace_raw', calcStat(gldas.twsnldas[n_p-1:nEnd], gldas.twsgrace[n_p-1:nEnd]))    
    print ('Training: predicted vs. grace_raw',calcStat(gldas.twsnldas[n_p-1:nTrain]-predAll[:nTrain-n_p+1], gldas.twsgrace[n_p-1:nTrain]))
    print ('Training: gldas vs. grace_raw', calcStat(gldas.twsnldas[n_p-1:nTrain], gldas.twsgrace[n_p-1:nTrain]))    
    print ('Testing: predicted vs. grace_raw', calcStat(gldas.twsnldas[nTrain:nEnd]-predAll[nTrain-n_p+1:nEnd-n_p+1], gldas.twsgrace[nTrain:nEnd])) 
    print ('Testing: gldas vs. grace_raw', calcStat(gldas.twsnldas[nTrain:nEnd], gldas.twsgrace[nTrain:nEnd]))
    print ('\n')
    print ('Indus Training: predicted vs. grace_raw',calcStat(gldas.twsnldas_indus[n_p-1:nTrain]-indusAll[:nTrain-n_p+1], gldas.twsgrace_indus[n_p-1:nTrain]))
    print ('Indus Testing: predicted vs. grace_raw', calcStat(gldas.twsnldas_indus[nTrain:nEnd]-indusAll[nTrain-n_p+1:nEnd-n_p+1], gldas.twsgrace_indus[nTrain:nEnd]))
    print ('\n')
    print ('Ganges Training: predicted vs. grace_raw', calcStat(gldas.twsnldas_ganges[n_p-1:nTrain]-gangesAll[:nTrain-n_p+1], gldas.twsgrace_ganges[n_p-1:nTrain]))
    print ('Ganges Testing: predicted vs. grace_raw', calcStat(gldas.twsnldas_ganges[nTrain:nEnd]-gangesAll[nTrain-n_p+1:nEnd-n_p+1], gldas.twsgrace_ganges[nTrain:nEnd]))
    
    print ('------------------------------------------------------------------------------------')

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
    labeldict={'simple':'(a)', 'vgg16':'(b)', 'compvggSRT':'(c)', 
               'ndvicnn':'(d)', 'ndvivgg':'e', 'unet':'g', "compcnn":'h', 
               "unetp":'i', "unetndviSRT":'j', 'climvgg':'k','ndvivggSRT':'1'}
    ax.text(0.02, 0.90, labeldict[label], fontsize=10, transform=ax.transAxes)
    plt.savefig('gldas_timeseriesplot%s.png' % label, dpi=fig.dpi)

    fig=plt.figure(figsize=(6,3), dpi=250)
    ax = plt.subplot(111)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    T0 = 2002+4.0/12
    rngTrain = T0+np.array(range(n_p-1,nTrain))/12.0
    rngTest = T0+np.array(range(nTrain, nEnd))/12.0
    rngFull = T0+np.array(range(n_p-1,nEnd))/12.0
    ax.plot(rngTrain, gldas.twsnldas_indus[n_p-1:nTrain]-indusAll[:nTrain-n_p+1], '--', color='#2980B9', label='CNN Train')
    ax.plot(rngTrain, gldas.twsgrace_indus[n_p-1:nTrain], '-', color='#7D3C98', label='GRACE Train')
    ax.plot(rngTest, gldas.twsnldas_indus[nTrain:nEnd]-indusAll[nTrain-n_p+1:nEnd-n_p+1], '--o', color='#2980B9', label='CNN Test',markersize=3.5)
    ax.plot(rngTest, gldas.twsgrace_indus[nTrain:nEnd], '-o', color='#7D3C98', label='GRACE Test', markersize=3.5)
    ax.plot(rngFull, gldas.twsnldas_indus[n_p-1:nEnd], ':', color='#626567', label='NOAH')
    ax.axvspan(xmin=T0+(nTrain-1.5)/12, xmax=T0+(nTrain+0.5)/12, facecolor='#7B7D7D', linewidth=2.0)
    ax.grid(False)
    ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.65), borderaxespad=0., fancybox=True, frameon=True)
    ax.text(0.02, 0.90, labeldict[label], fontsize=10, transform=ax.transAxes)
    plt.savefig('gldasindus_timeseriesplot%s.png' % label, dpi=fig.dpi)

    fig=plt.figure(figsize=(6,3), dpi=250)
    ax = plt.subplot(111)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    T0 = 2002+4.0/12
    rngTrain = T0+np.array(range(n_p-1,nTrain))/12.0
    rngTest = T0+np.array(range(nTrain, nEnd))/12.0
    rngFull = T0+np.array(range(n_p-1,nEnd))/12.0
    ax.plot(rngTrain, gldas.twsnldas_ganges[n_p-1:nTrain]-gangesAll[:nTrain-n_p+1], '--', color='#2980B9', label='CNN Train')
    ax.plot(rngTrain, gldas.twsgrace_ganges[n_p-1:nTrain], '-', color='#7D3C98', label='GRACE Train')
    ax.plot(rngTest, gldas.twsnldas_ganges[nTrain:nEnd]-gangesAll[nTrain-n_p+1:nEnd-n_p+1], '--o', color='#2980B9', label='CNN Test',markersize=3.5)
    ax.plot(rngTest, gldas.twsgrace_ganges[nTrain:nEnd], '-o', color='#7D3C98', label='GRACE Test', markersize=3.5)
    ax.plot(rngFull, gldas.twsnldas_ganges[n_p-1:nEnd], ':', color='#626567', label='NOAH')
    ax.axvspan(xmin=T0+(nTrain-1.5)/12, xmax=T0+(nTrain+0.5)/12, facecolor='#7B7D7D', linewidth=2.0)
    ax.grid(False)
    ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.65), borderaxespad=0., fancybox=True, frameon=True)
    ax.text(0.02, 0.90, labeldict[label], fontsize=10, transform=ax.transAxes)
    plt.savefig('gldasganges_timeseriesplot%s.png' % label, dpi=fig.dpi)

    #save results for plotting
    if label == 'ndvivggSRT' or label=='compvggSRT' or label=='unetndviSRT':
        outname = 'basincorr{0}{1}.pkl'.format(label, nrz)
    else:
        outname = 'basincorr{0}.pkl'.format(label)

    #save results for plotting
    pkl.dump([nTrain,rngTrain,rngTest, rngFull, gldas.twsnldas[n_p-1:nTrain]-predAll[:nTrain-n_p+1],
              gldas.twsgrace[n_p-1:nTrain],
              gldas.twsnldas[nTrain:nEnd]-predAll[nTrain-n_p+1:nEnd-n_p+1],
              gldas.twsgrace[nTrain:nEnd],
              gldas.twsnldas[n_p-1:nEnd]], open(outname, 'wb'))

    #plotOutput(nldas, ypred, Y_test, X_test)
    #plot correlation matrix
    plotCorrmat(label, gldas, corrmat)
    plotNSEmat(label, gldas, nsemat)
    #
    if doWellcorrel:
        plotGwCorrmat(label, gldas, gwcorrmat, True)
        plotGwCorrmatWell(label, gldas, wellcorrmat)

def main(retrain=False, modelnum=5, modeloption=None, realizationNo=''):
    watershed='indiabig'
    watershedActual = 'indiabang'
    n_p = 3 
    nTrain = 125
    if modelnum==2:
        CNNDriver(watershedName=watershed, watershedInner=watershedActual, 
                  n_p=n_p, nTrain=nTrain, retrain=retrain, modelOption=modeloption)
    elif modelnum==3:
        CNNCompositeDriver(watershedName=watershed, watershedInner=watershedActual, 
                           n_p=n_p, nTrain=nTrain, retrain=retrain, modelOption=modeloption,
                           realizationNum=realizationNo)
    elif modelnum==5:
        CNNNDVICompositeDriver(watershedName=watershed, watershedInner=watershedActual, 
                               retrain=retrain, modelOption=modeloption, realizationNum=realizationNo)
    elif modelnum==4:
        UnetDriver(watershedName=watershed, watershedInner=watershedActual, 
                   retrain=retrain, modeloption=modeloption, nrz=realizationNo)

def runRealization(seednum, irz):
        print ('seed set to %s ' % seed)
        np.random.seed(seednum)
        tf.set_random_seed(seednum)
        #realization 0 - 9
        #np.random.seed(1111)

        server = tf.train.Server.create_local_server()
        sess = tf.Session(server.target)    
        K.set_session(sess)
        with sess.as_default():
            #main(retrain=True, modelnum=3, modeloption=2, realizationNo=irz)
            #main(retrain=True, modelnum=5, modeloption=2, realizationNo=irz)
            main(retrain=True, modelnum=4, modeloption=3, realizationNo=irz)
        sess.close()

if __name__ == "__main__":
    np.random.seed(1989)
    seeds = np.random.randint(1001, 9999, size=20)
    for i in range(10, 20):
        runRealization(seeds[i], i)
