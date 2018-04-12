# -*- coding: utf-8 -*-
import keras
from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Flatten,Activation,Reshape,Masking
from keras.layers.convolutional import Conv3D, Conv2D, MaxPooling3D, MaxPooling2D, ZeroPadding3D
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, BatchNormalization, merge, TimeDistributed, LSTM, ConvLSTM2D, UpSampling2D, merge  
from keras.layers import Convolution2D, AtrousConvolution2D
from keras.losses import categorical_crossentropy
from keras.models import load_model
from keras.optimizers import SGD, Adam,RMSprop
from keras.utils import plot_model

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

def seqCNN1(seq_len=6, summary=False,backend='tf', N=64):
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

def seqCNN3(seq_len=3, summary=False,backend='tf', N=64):
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
def vcg16CNN(seq_len=6, summary=False,backend='tf', N=64):
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
def resnetCNN(seq_len=6, summary=False, backend='tf', N=64):
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

def unetModel(Ns, seq_len=3, summary=False, N=64):

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

def lstmModel(Ns, seq_len=3, backend='tf', N=64):
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

def compositeLSTM(n_p=None, summary=False, backend='tf', N=64):
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
    model = lstmModel(Ns=N, seq_len=n_p, N=N)
    if summary:
        print(model.summary())
    return model

def compositeLSTM2(n_p=None, summary=False, backend='tf', N=64):
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
    
    if summary:
        print(model.summary())
        
    return model