#author: alex sun
#05022018,modified for prediction outside validation area
#This code requires the modified gldasBasinPredict.py
#To run, use option base to generate base result 
#and then use option mc to run
#the results are in  gldasBasinPredict.PREDICT_DIR
#=====================================================================================
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
import keras
from keras.optimizers import SGD, Adam,RMSprop

from keras.layers import Input, BatchNormalization,ConvLSTM2D, UpSampling2D  
from keras.losses import categorical_crossentropy
from keras.models import load_model
import sys
import keras.backend as K
from keras import regularizers
import seaborn as sns
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from gldasBasinPredict import GLDASData, NDVIData, PREDICT_DIR
import gldasBasinPredict

from keras.callbacks import ReduceLROnPlateau,EarlyStopping

K.set_session(sess)
from keras.utils import plot_model

from scipy.stats.stats import pearsonr
import sklearn.linear_model as skpl
import pandas as pd
from ProcessIndiaDB import ProcessIndiaDB
from numpy.random import seed
seed(1111)
import pickle as pkl

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
N=gldasBasinPredict.N
MASKVAL=np.NaN
PREDICT_DIR = gldasBasinPredict.PREDICT_DIR

from keras.applications.vgg16 import VGG16
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

def CNNNDVICompositeUQ(watershedName, watershedInner, n_p=3, nTrain=125, 
                           modelOption=1, retrain=False):
    '''
    This  is the VGG16-3 model mentioned in the paper
    '''
    isMasking=True
    nldas = GLDASData(watershed=watershedName,watershedInner=watershedInner)
    nldas.loadStudyData(reloadData=False,masking=isMasking)    
    ndvi = NDVIData(reLoad=False, watershed=watershedName) 
    
    X_train, X_test = nldas.formMatrix2D(n_p=n_p, masking=isMasking, nTrain=nTrain)
    Xp_train, Xp_test = nldas.formPrecip2D(n_p=n_p, masking=isMasking, nTrain=nTrain)
    Xnd_train, Xnd_test = nldas.formNDVI2D(ndvi, n_p=n_p, masking=isMasking, nTrain=nTrain)

    backend='tf'
    if backend == 'tf':
        input_shape=(N, N, n_p) # l, h, w, c
    else:
        input_shape=(n_p, N, N) # c, l, h, w

    if modelOption==2:
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
    nrz = 20
    results = None
    for irz in range(nrz):
        model.load_weights('gloindiabigndvimodel_weightsndvivggSRT{0}.h5'.format(irz))
        print ('loaded vgg16 model ', irz )
        tmp,_ = doTesting(model, label, nldas, X_train, X_test, Xp_train, Xp_test,
          Xnd_train, Xnd_test, n_p=n_p, nTrain=nTrain, pixel=False)

        if results is None:
            results = np.zeros((nrz, len(tmp)))
        results[irz,:] = tmp

    pkl.dump(results, open('{0}/result.pkl'.format(PREDICT_DIR), 'wb')) 
        

def CNNNDVICompositeDriver(watershedName, watershedInner, n_p=3, nTrain=125, 
                           modelOption=1, retrain=False):
    '''
    This  is the VGG16-3 model mentioned in the paper
    '''
    isMasking=True
    nldas = GLDASData(watershed=watershedName,watershedInner=watershedInner)
    nldas.loadStudyData(reloadData=False,masking=isMasking)    
    ndvi = NDVIData(reLoad=False, watershed=watershedName) 
    
    X_train, X_test = nldas.formMatrix2D(n_p=n_p, masking=isMasking, nTrain=nTrain)
    Xp_train, Xp_test = nldas.formPrecip2D(n_p=n_p, masking=isMasking, nTrain=nTrain)
    Xnd_train, Xnd_test = nldas.formNDVI2D(ndvi, n_p=n_p, masking=isMasking, nTrain=nTrain)

    backend='tf'
    if backend == 'tf':
        input_shape=(N, N, n_p) # l, h, w, c
    else:
        input_shape=(n_p, N, N) # c, l, h, w

    if modelOption==2:
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
    
    model.load_weights('glo{0}ndvimodel_weights{1}SRT.h5'.format(watershedName,label))
    print ('loaded vgg16 model')
    res, _ = doTesting(model, label, nldas, X_train, X_test, Xp_train, Xp_test,
              Xnd_train, Xnd_test, n_p=n_p, nTrain=nTrain, pixel=False)
    pkl.dump(res, open('{0}/base.pkl'.format(PREDICT_DIR), 'wb'))

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
    
def UnetDriver(watershedName, watershedInner, retrain=False, n_p=3, nTrain=125, modeloption=1):
    isMasking=True
    
    nldas = GLDASData(watershed=watershedName,watershedInner=watershedInner)
    nldas.loadStudyData(reloadData=False)    
    ndvi = NDVIData(reLoad=False, watershed=watershedName)
    
    input_shape = (N,N,n_p)

    X_train, X_test = nldas.formMatrix2D(n_p=n_p, masking=isMasking, nTrain=nTrain)
    Xp_train, Xp_test = nldas.formPrecip2D(n_p=n_p, masking=isMasking, nTrain=nTrain)
    Xnd_train, Xnd_test = nldas.formNDVI2D(ndvi, n_p=n_p, masking=isMasking, nTrain=nTrain)
    
    if modeloption==3:
        inLayer = Input(shape=input_shape, name='input')
        outLayer = Conv2D(16, kernel_size=(3, 3), padding='same', activation='relu')(inLayer)
            
        inputPLayer = Input(shape=input_shape, name='inputP')
        outPLayer = Conv2D(16, kernel_size=(3, 3), padding='same',activation='relu')(inputPLayer)

        inputNDVILayer = Input(shape=(N,N,n_p), name='inputNDVI')
        outNDVILayer = Conv2D(16, kernel_size=(3, 3), padding='same',activation='relu')(inputNDVILayer)

        x = keras.layers.concatenate([outPLayer, outNDVILayer, outLayer], axis=-1)  
        x = getUnetModel(n_p, summary=True, inputLayer=x)
        label = 'unetndvi'
 
        model = Model(inputs=[inputPLayer, inputNDVILayer, inLayer], outputs=[x])

    weightfile='{0}{1}model_weights.h5'.format(watershedName,label)
    model.load_weights(weightfile)
    
    res, noah = doTesting(model, label, nldas, X_train, X_test, Xp_train, Xp_test,
              Xnd_train, Xnd_test, n_p=n_p, nTrain=nTrain, pixel=False)

    pkl.dump([res, noah], open('{0}/base.pkl'.format(PREDICT_DIR), 'wb'))   

def UnetUQ(watershedName, watershedInner, retrain=False, n_p=3, nTrain=125, modeloption=1):
    isMasking=True
    
    nldas = GLDASData(watershed=watershedName,watershedInner=watershedInner)
    nldas.loadStudyData(reloadData=False)    
    ndvi = NDVIData(reLoad=False, watershed=watershedName)
    
    input_shape = (N,N,n_p)

    X_train, X_test = nldas.formMatrix2D(n_p=n_p, masking=isMasking, nTrain=nTrain)
    Xp_train, Xp_test = nldas.formPrecip2D(n_p=n_p, masking=isMasking, nTrain=nTrain)
    Xnd_train, Xnd_test = nldas.formNDVI2D(ndvi, n_p=n_p, masking=isMasking, nTrain=nTrain)
    
        
    if modeloption==3:
        inLayer = Input(shape=input_shape, name='input')
        outLayer = Conv2D(16, kernel_size=(3, 3), padding='same', activation='relu')(inLayer)
            
        inputPLayer = Input(shape=input_shape, name='inputP')
        outPLayer = Conv2D(16, kernel_size=(3, 3), padding='same',activation='relu')(inputPLayer)

        inputNDVILayer = Input(shape=(N,N,n_p), name='inputNDVI')
        outNDVILayer = Conv2D(16, kernel_size=(3, 3), padding='same',activation='relu')(inputNDVILayer)

        x = keras.layers.concatenate([outPLayer, outNDVILayer, outLayer], axis=-1)  
        x = getUnetModel(n_p, summary=True, inputLayer=x)
        label = 'unetndviSRT'
 
        model = Model(inputs=[inputPLayer, inputNDVILayer, inLayer], outputs=[x])

    nrz = 20
    results = None
    for irz in range(nrz):
        model.load_weights('{0}{1}model_weightsunet{2}.h5'.format(watershedName, label, irz))
        print ('loaded unet model ', irz) 
        tmp, _ =     doTesting(model, label, nldas, X_train, X_test, Xp_train, Xp_test,
              Xnd_train, Xnd_test, n_p=n_p, nTrain=nTrain, pixel=False)

        if results is None:
            results = np.zeros((nrz, len(tmp)))
        results[irz,:] = tmp
        print (tmp)

    pkl.dump(results, open('{0}/result.pkl'.format(PREDICT_DIR), 'wb')) 
    

def backTransform(nldas, mat):
    '''
    nldas, an instance of the nldas class
    '''
    temp = nldas.outScaler.inverse_transform((mat[nldas.validCells]).reshape(1, -1))
    res = np.zeros((N,N))
    res[nldas.validCells] = temp
    return res

def calculateBasinAverage(nldas, Y):
    '''
    Y is either predicted or test tensor
    '''

    nvalidCell = nldas.nActualCells
    mask = nldas.innermask
    
    tws_avg = np.zeros((Y.shape[0]))
    for i in range(Y.shape[0]):
        obj = backTransform(nldas, np.multiply(Y[i,:,:], mask))
        tws_avg[i] = np.nansum(obj)/nvalidCell
    return tws_avg

def doTesting(model, label, gldas, X_train, X_test, Xa_train=None, Xa_test=None, 
              Xnd_train=None, Xnd_test=None, n_p=3, nTrain=106, pixel=False):
    
    def calcSubbasinTWS():
        #if basinname is None, the whole India will be returned
        twsTrain = calculateBasinAverage(gldas, ytrain)    
        twsPred = calculateBasinAverage(gldas, ypred)        
        #calculate correlation
        predAll = np.r_[twsTrain, twsPred]
        return predAll
    #load outscaler saved in gldasBasin.py
    gldas.outScaler = pkl.load(open('outscaler.pkl', 'rb'))
    #testing data
    nEnd = 189-n_p
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
    
    predAll = calcSubbasinTWS()
    
    #plotTWS(gldas, label, predAll, nTrain, nEnd, n_p)
    return gldas.twsnldas[n_p-1:nEnd]-predAll[:nEnd-n_p+1], gldas.twsnldas[n_p-1:nEnd]

def plotTWS(gldas, label, predAll, nTrain, nEnd, n_p):
    sns.set_style("white")
    fig=plt.figure(figsize=(6,3), dpi=250)
    ax = plt.subplot(111)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    T0 = 2002+4.0/12
    #176, 177
    print ('nldas is ', gldas.twsnldas)
    print ('predall is ', predAll)
    rngTrain = T0+np.array(range(n_p-1,nTrain))/12.0
    rngTest = T0+np.array(range(nTrain, nEnd))/12.0
    rngFull = T0+np.array(range(n_p-1,nEnd))/12.0
    ax.plot(rngTrain, gldas.twsnldas[n_p-1:nTrain]-predAll[:nTrain-n_p+1], '--', color='#2980B9', label='CNN Train')
    ax.plot(rngTest, gldas.twsnldas[nTrain:nEnd]-predAll[nTrain-n_p+1:nEnd-n_p+1], '--o', color='#2980B9', label='CNN Test',markersize=3.5)
    ax.plot(rngFull, gldas.twsnldas[n_p-1:nEnd], ':', color='#626567', label='NOAH')
    ax.axvspan(xmin=T0+(nTrain-1.5)/12, xmax=T0+(nTrain+0.5)/12, facecolor='#7B7D7D', linewidth=2.0)
    ax.grid(True)
    ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.65), borderaxespad=0., fancybox=True, frameon=True)
    ax.set_xlim(2002,2018)
    plt.savefig('{0}/gldaspredict_timeseriesplot{1}.png'.format(PREDICT_DIR, label), dpi=fig.dpi)
    
def main():
    '''
    this is for trained vgg16-3 model only
    '''
    watershed='indiabig'
    watershedActual = 'indiabang'
    n_p = 3 
    nTrain = 125
    retrain = False
    runOption = 'base' #valid options are base and mc
    if runOption == 'base':
        #this is for vgg16-3
        #CNNNDVICompositeDriver(watershedName=watershed, watershedInner=watershedActual, 
        #                   retrain=retrain, modelOption=modeloption)
        #this is for unet-3
        UnetDriver(watershedName=watershed, watershedInner=watershedActual, 
                   retrain=retrain, modeloption=3)        
    else:
        UnetUQ(watershedName=watershed, watershedInner=watershedActual, 
                   retrain=retrain, modeloption=3)        

if __name__ == "__main__":
    main()