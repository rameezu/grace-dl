#author: Alex Sun
#date: 1/21/2018
#purpose: do nldas to grace prediction for Colorado
#=====================================================================================
import numpy as np
import matplotlib
matplotlib.use('Agg')

from netCDF4 import Dataset
import pandas as pd
import sys,os
import matplotlib.pyplot as plt

from datetime import datetime, timedelta,date
import calendar
from skimage.transform import resize
from scipy.misc import imresize
from scipy import stats
import sklearn.preprocessing as skp
from matplotlib import path
import seaborn as sns
from matplotlib.colors import ListedColormap
import keras.backend as K

import warnings
warnings.filterwarnings('ignore')

#define nldas information
xllcorner =-124.9375
yllcorner =25.0625
cellsize =0.125
nrows = 224
ncols = 464
varDict = {'precip':'apcpsfc', 'temp':'tmp2m', 'sm200':'soilm0_200cm',  
           'runoff':'ssrunsfc', 'snow':'weasdsfc', 'canopy': 'cnwatsfc'}

N = 64
MASKVAL=0.0
RESIZE =2 #set 1 to use sciki-image, 2 to use imresize
XMIN = -0.99 #these are for scaling the images
XMAX =  0.99  
cmap = ListedColormap((sns.color_palette("RdBu", 10)).as_hex())

class GRACEData():
    '''
    load grace data,
    Current date from 2002/04 to 2017/06
    '''
    def __init__(self, watershed):        
        self.twsData, self.nmonths, self.monthind, self.extents = np.load('grace{0}.npy'.format(watershed), encoding='latin1')
        #make study period = 2002/04 to 2016/12               
        self.twsData = self.twsData[:-6,:]
            

class NLDASData():
    '''
    purpose: get tws simulated by nldas, data range 1979/1/1 to 2016/12/31 
    '''
    def __init__(self, watershed, startYear=1979, endYear=2016):
        '''
        watershed, name of the watershed file
        '''
        self.dataroot = '.'
        self.startYear = startYear
        self.endYear = endYear
        self.watershed = watershed
        self.nldastws = None
        self.inScaler = None
        self.outScaler = None
        
        self.precipMat= np.load('colorado_precipdata.npy')
        self.precipMat= self.precipMat[(2002-1979)*12+3:,:] #this retrieves for the study period
        
    def loadStudyData(self, reloadData=False):
        def diff_month(d2, d1):
            #returns elapsed months between two dates
            return (d1.year - d2.year) * 12 + d1.month - d2.month +1
        def calcMonthAvg(mat):
            #calculates monthly averages
            nmonths = diff_month(datetime(self.startYear,1,1), datetime(self.endYear,12,1))
            monmat = np.zeros((nmonths, mat.shape[1]))
            counter=0
            days = 0
            for iyear in range(self.startYear, self.endYear+1):
                for imon in range(1,13):
                    monthdays = calendar.monthrange(iyear,imon)[1]
                    monmat[counter,:] = np.mean(mat[days:days+monthdays,:],axis=0, dtype='float64')
                    days+=monthdays                    
                    counter+=1
            return monmat
        
        bryfile = '{0}/bdy/{1}.bdy'.format(self.dataroot, self.watershed)
        
        extents = self.getBasinPixels(bryfile)    
        print(('basin extents is ', extents))
        self.mask = self.generateMask(bryfile)    
        #extract cells inside mask
        self.validCells = np.where(self.mask==1)
        self.nvalidCells = len(self.validCells[0])
        '''
        #for debugging
        plt.figure()
        plt.imshow(self.mask)
        plt.savefig('nldasmask.png')
        '''
        self.nldastwsB, self.precipMat = np.load('{0}_twsdata.npy'.format(self.watershed), encoding='latin1')
        self.extents = extents
        #
        #align nldas data with grace period
        #from 2002/04 to 2016/12 (note I need to use +3 because the zero-based index
        self.nldastws = self.nldastwsB[(2002-1979)*12+3:,:]
        print((self.nldastws.shape))
        
    
    def getBasinPixels(self, bryfile):
        with open(bryfile) as f:
            line = (f.readline()).strip()
            fields = line.split()
            nlines = int(fields[1])        

            locx = np.zeros((nlines,1))
            locy = np.zeros((nlines,1))
            counter=0
            for line in f:                                
                fields = line.strip().split()
                locx[counter] = float(fields[1])
                locy[counter] = float(fields[0])        
                counter+=1

        indx = np.ceil((locx-xllcorner)/cellsize)
        indy = np.ceil((locy-yllcorner)/cellsize)
        return int(np.min(indx))-1, int(np.max(indx)), int(np.min(indy))-1, int(np.max(indy))

    def generateMask(self, bryfile):
        with open(bryfile) as f:
            line = (f.readline()).strip()
            fields = line.split()
            nlines = int(fields[1])        

            locx = np.zeros((nlines,1))
            locy = np.zeros((nlines,1))
            counter=0
            for line in f:                                
                fields = line.strip().split()
                locx[counter] = float(fields[1])
                locy[counter] = float(fields[0])        
                counter+=1
                
        locx = locx-xllcorner
        locy = locy-yllcorner
        vertices = np.hstack((locx.flatten()[:,np.newaxis], locy.flatten()[:,np.newaxis]))
        p = path.Path(vertices)          

        indx = np.ceil(locx/cellsize)
        indy = np.ceil(locy/cellsize)
        
        ix0,ix1=int(np.max(indx)), int(np.min(indx))
        nX = ix0-ix1
        iy0,iy1=int(np.max(indy)), int(np.min(indy))
        nY = iy0-iy1
        dx = (np.max(locx) - np.min(locx))/N
        dy = (np.max(locy) - np.min(locy))/N

        xv,yv = np.meshgrid(np.linspace((ix1-1)*cellsize+0.5*dx,(ix0)*cellsize-0.5*dx, N),
                            np.linspace((iy1-1)*cellsize+0.5*dy,(iy0)*cellsize-0.5*dy, N))

        flags = p.contains_points(np.hstack((xv.flatten()[:,np.newaxis],yv.flatten()[:,np.newaxis])))        

        basinMask = np.zeros(flags.shape, dtype='float')+1
        basinMask[flags==False]=MASKVAL

        return np.reshape(basinMask, (N,N))

    def formPrecip2D(self, n_p=3, masking=False, nTrain=106):
    
        '''
        Theano: (nsamples, times, channel, height, width)
        Tensorflow:  (nsamples, times, height, width, channel) <<used here>>
        n_p, the number of antecedent states to use
        @param n_p, maximum lag
        '''           
    
        if (self.precipMat is None):
            raise Exception('must call loadstudyarea first')
        mat = self.precipMat
        self.nmonths = mat.shape[0]       
    
        #set up matrices for conv layers
        self.inArr = self.formatInArray(mat, masking, varname='precip')
    
        '''
        #debugging
        plt.figure()
        im = plt.imshow(self.inArr[0,:,:], cmap)
        plt.colorbar(im)
        plt.savefig('testprecip.png')
        sys.exit()
        '''
        if K.image_data_format()=="channels_first": #teano
            Xtrain = np.zeros((nTrain-n_p, n_p, N, N), dtype=np.float64)           
    
            for i in range(n_p, nTrain):
                Xtrain[i-n_p, :, :, :] = self.inArr[i-n_p:i, :, :]               
    
            Xtest = np.zeros((self.nmonths-nTrain, n_p, N, N), dtype=np.float64)
         
            for i in range(nTrain, self.nmonths):
                Xtest[i-nTrain, :,  :, :] = self.inArr[i-n_p:i, :, :]
    
        else:
            Xtrain = np.zeros((nTrain-n_p+1, N, N, n_p), dtype=np.float64)
    
            for i in range(n_p-1, nTrain):
                for j in range(1,n_p+1):
                    Xtrain[i-n_p+1, :, :, j-1] = self.inArr[i-n_p+j, :, :]
     
            Xtest = np.zeros((self.nmonths-nTrain+1, N, N, n_p), dtype=np.float64)           
    
            for i in range(nTrain, self.nmonths):
                for j in range(1,n_p+1):
                    Xtest[i-nTrain, :, :, j-1] = self.inArr[i-n_p+j, :, :]
    
        return Xtrain,Xtest

    def formatInArray(self, arr, masking, varname='nldas'):
        '''
        format an array 
        '''
        bigarr = np.zeros((arr.shape[0], self.nvalidCells), dtype=np.float64)
        for i in range(arr.shape[0]):
            img0 = np.reshape(arr[i,:], (self.extents[3]-self.extents[2], 
                                self.extents[1]-self.extents[0]))                          
            #normalizing data to the range [0, 1]
            #img0 /= maxval
            img0[np.isnan(img0)] = 0.0
            #02012018, I checked that the mass balance conserved after image resizing
            #print 'before', np.sum(img0)
            if RESIZE==1:
                res = np.array(resize(img0, output_shape=(N,N), preserve_range=True), dtype=np.float64)
            else:
                res = np.array(imresize(img0, size=(N,N), mode='F', interp='bilinear'), dtype=np.float64)

            if masking:
                res=np.multiply(res,self.mask)
            
            bigarr[i,:] = res[self.validCells]
        if varname=='nldas':
            self.inScaler = skp.MinMaxScaler(feature_range=(XMIN, XMAX), copy=False)
            self.inScaler.fit_transform(bigarr)
        elif varname == 'precip':
            self.pScaler = skp.MinMaxScaler(feature_range=(XMIN, XMAX), copy=False)
            self.pScaler.fit_transform(bigarr)
            
        resarr = np.zeros((bigarr.shape[0], N, N))

        for i in range(bigarr.shape[0]):
            temp = np.zeros((N,N))
            temp[self.validCells] = bigarr[i,:]
            resarr[i,:,:] = temp
        return resarr
        
    def formatOutArray(self, arr1, arr2, extents2, masking):
        '''
        format output array
        '''
        bigarr = np.zeros((arr1.shape[0], self.nvalidCells), dtype=np.float64)
        for i in range(arr1.shape[0]):
            img0 = np.reshape(arr1[i,:], (self.extents[3]-self.extents[2], 
                                self.extents[1]-self.extents[0]))              
            img1 = np.reshape(arr2[i,:], (extents2[3]-extents2[2],extents2[1]-extents2[0])) 
                            
            #normalizing data to the range [0, 1]
            #img0 /= maxval
            if RESIZE==1:
                d1 = np.array(resize(img0, output_shape=(N,N),preserve_range=True), dtype=np.float64)
                d2 = np.array(resize(img1, output_shape=(N,N),preserve_range=True), dtype=np.float64)
                res = d1-d2
            else:
                d1 = np.array(imresize(img0, size=(N,N), mode='F', interp='bilinear'), dtype=np.float64)
                d2 = np.array(imresize(img1, size=(N,N), mode='F', interp='bilinear'), dtype=np.float64) 
                res = d1-d2
                
            
            #for debugging, print out grace space average
            if i==10:               
                plt.figure(figsize=(12,6))
                plt.subplot(1,3,1)            
                d1[np.where(self.mask==0)]=np.NaN
                im = plt.imshow(d1, cmap)
                plt.colorbar(im,  orientation="horizontal", fraction=0.046, pad=0.1)   
                plt.title('NLDAS/NOAH')
                
                plt.subplot(1,3,2)            
                d2[np.where(self.mask==0)]=np.NaN
                im = plt.imshow(d2, cmap)
                plt.colorbar(im, orientation="horizontal", fraction=0.046, pad=0.1)   
                plt.title('GRACE')
                
                plt.subplot(1,3,3)
                res[np.where(self.mask==0)]=np.NaN
                im = plt.imshow(res, cmap)
                plt.title('NLDAS-GRACE')
                plt.colorbar(im, orientation="horizontal", fraction=0.046, pad=0.1)
                plt.savefig('nldas-gracetest.png')
                
            if masking:
                res = np.multiply(res,self.mask)
                        
            bigarr[i,:] = res[self.validCells]

        self.outScaler = skp.MinMaxScaler(feature_range=(XMIN, XMAX), copy=False)
        self.outScaler.fit_transform(bigarr)
        resarr = np.zeros((bigarr.shape[0], N, N))

        for i in range(bigarr.shape[0]):
            temp = np.zeros((N,N))
            temp[self.validCells] = bigarr[i,:]
            resarr[i,:,:] = temp
        return resarr

    def getTWSAvg(self, mat, extents):
        '''
        calculate average tws for both grace and nldas
        '''
        twsavg = np.zeros((mat.shape[0]))
        for i in range(mat.shape[0]):
            img = np.reshape(mat[i,:], (extents[3]-extents[2],extents[1]-extents[0])) 
            res = imresize(img, size=(N,N), mode='F', interp='bilinear')
            res = np.multiply(res,self.mask)
            twsavg[i] = np.sum(res)/self.nvalidCells
        return twsavg
    
    def formMatrix2D(self, gl, n_p=3, masking=False, nTrain=1000):
        '''
        Theano: (nsamples, times, channel, height, width)
        Tensorflow:  (nsamples, times, height, width, channel) <<used here>>
        @param gl = instance of GRACE (obtained using GRACE  class) 
        @param n_p, the number of antecedent states to use, t, t-1, ...
        @param masking, set to True to extract the watershed pixels
        @param nTrain: number of training samples
        '''            
        if self.nldastws is None:
            raise Exception('must loadstudyarea first')
        if gl is None:
            raise Exception('grace tws cannot be none')
        
        mat = self.nldastws        
        self.nmonths = mat.shape[0]
        gmat = gl.twsData
        gextents = gl.extents
        
        #set up matrices for conv layers
        self.inArr = self.formatInArray(mat, masking)
        self.outArr = self.formatOutArray(mat, gmat, gextents, masking)
        #
        self.twsgrace = self.getTWSAvg(gmat, gextents)
        self.twsnldas = self.getTWSAvg(mat, self.extents)
        
        if K.image_data_format()=="channels_first": #teano 
            Xtrain = np.zeros((nTrain-n_p, n_p, N, N), dtype=np.float64)            
            Ytrain = np.zeros((nTrain-n_p, N, N), dtype=np.float64)
            
            for i in range(n_p, nTrain):
                Xtrain[i-n_p, :, :, :] = self.inArr[i-n_p:i, :, :]                
                Ytrain[i-n_p, :, :] = self.outArr[i, :, :]
                
            Xtest = np.zeros((self.nmonths-nTrain, n_p, N, N), dtype=np.float64)
            Xval =  np.zeros((self.nmonths-nTrain, N, N), dtype=np.float64)
            Ytest = np.zeros((self.nmonths-nTrain, N, N), dtype=np.float64)
            
            for i in range(nTrain, self.nmonths):
                Xtest[i-nTrain, :,  :, :] = self.inArr[i-n_p:i, :, :]
                Ytest[i-nTrain, :, :] = self.outArr[i, :, :]
                Xval[i-nTrain, :, :] = self.inArr[i,:,:]
        else: 
            Xtrain = np.zeros((nTrain-n_p+1, N, N, n_p), dtype=np.float64)
            Ytrain = np.zeros((nTrain-n_p+1, N, N), dtype=np.float64)
            
            for i in range(n_p-1, nTrain):
                for j in range(1,n_p+1):
                    Xtrain[i-n_p+1, :, :, j-1] = self.inArr[i-n_p+j, :, :]
                Ytrain[i-n_p+1, :,:] = self.outArr[i,:,:]
                
            Xtest = np.zeros((self.nmonths-nTrain+1, N, N, n_p), dtype=np.float64)
            Ytest = np.zeros((self.nmonths-nTrain+1, N, N), dtype=np.float64)
            Xval =  np.zeros((self.nmonths-nTrain+1, N, N), dtype=np.float64)
            
            for i in range(nTrain, self.nmonths):
                for j in range(1,n_p+1):
                    Xtest[i-nTrain, :, :, j-1] = self.inArr[i-n_p+j, :, :]
                Ytest[i-nTrain, :, :] = self.outArr[i,:,:]
                Xval[i-nTrain, :, :] =  self.inArr[i,:,:]
                
        return Xtrain,Ytrain,Xtest,Ytest,Xval        
        
def main():
    '''
    right now, data only works for watershed = colorado
    '''
    grace = GRACEData(watershed='colorado')    
    nldas = NLDASData(watershed='colorado')
    nldas.loadStudyData(reloadData=False)    
    Xtrain,Ytrain,Xtest,Ytest,Xval = nldas.formMatrix2D(gl=grace, n_p=3, masking=True, nTrain=140)
    print((Xtrain.shape))
if __name__ == "__main__":
    main()