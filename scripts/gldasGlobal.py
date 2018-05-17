#author: alex sun
#date: 1/21/2018
#purpose: do nldas to grace prediction
#2/10/2018 do data augmentation
#3/6/2018 adapted for use with GLDAS Global
#=====================================================================================
import numpy as np
import matplotlib
matplotlib.use('Agg')

from netCDF4 import Dataset
import pandas as pd
import sys,os
import matplotlib.pyplot as plt

from datetime import datetime, timedelta,date
from skimage.transform import resize
from scipy.misc import imresize
from scipy import stats
import sklearn.preprocessing as skp
from joblib import Parallel, delayed, load, dump
import tempfile
from matplotlib import path
import seaborn as sns
from matplotlib.colors import ListedColormap
import keras.backend as K
from keras.preprocessing.image import ImageDataGenerator

import imgaug as ia
from imgaug import augmenters as iaa


from PIL import Image
from ProcessIndiaDB import ProcessIndiaDB

#define GLDAS information
xllcorner =-179.875
yllcorner =-59.875
cellsize =0.25
nrows = 600
ncols = 1440

varDict = {'precip':'rainf_f_tavg', 'sm200':'sm100_200cm_ins',  
           'snow':'swe_inst', 'canopy': 'canopint_inst'}

N = 125
MASKVAL=0.0
RESIZE =2 #1 for sciki-image, 2 for scipy imresize
#valid choices, 'blinear', 'cubic', 'nearest'
#as02282018, it seems smoothing helps cnn significantly
INT_METHOD='cubic' 
#these are for linear scaling
XMIN = -0.99
XMAX =  0.99  
#
cmap = ListedColormap((sns.color_palette("RdBu", 10)).as_hex())
MISSINGVAL=-9999.0

def getGLData(pMat, iyear, imon, dataRoot, varname, extents, ncells):
    '''
    Get GLDAS data in a parallel way
    @param iyear, imon, month and year of the data file
    @param dataRoot, root of gldas data folder
    @param varname, name of the gldas variable
    @param extents, extents of watershed rect
    @param ncells, number of gldas cells 
    '''
    ncfile = ''.join([dataRoot, '/%s/%s%4d_%02d.nc' % (varname, varname, iyear, imon)])
    with Dataset(ncfile, mode='r') as fh:            
        lons = np.array(fh.variables['lon'][:])
        lats = np.array(fh.variables['lat'][:])
        arr = np.array( fh.variables[varDict[varname]][:])
        if varname=='precip':
            #convert average total hourly precip from (kg/m^2/s) to total daily (mm/day)
            # 1 kg/m^2 = 1 mm of H2O
            arr = arr*86400.0

        res = np.reshape(arr, (len(lats), len(lons)))
            
    res[res<=MISSINGVAL] = np.NaN

    res = res[extents[2]:extents[3], extents[0]:extents[1]]
    res = np.reshape(res, (1, ncells))
    ind = (iyear-2000)*12+imon-1        
    pMat[ind,:] = res

class GRACEData():
    '''
    Load grace data,
    Dataset includes 2002/04 to 2017/06 data
    '''
    def __init__(self, watershed, reLoad=False):        
        if reLoad:
            scalefile= '../nldas2/grace/CLM4.SCALE_FACTOR.JPL.MSCNv01CRIv01.nc'
            twsfile= '../nldas2/grace/GRCTellus.JPL.200204_201706.GLO.RL05M_1.MSCNv02CRIv02.nc'
            maskfile= '../nldas2/grace/LAND_MASK.CRIv01.nc'
            bryfile = './bdy/%s.bdy' % (watershed)
            self.extents = self.getExtent(bryfile)
            self.twsData, self.nmonths, self.monthind = self.loadTWS(twsfile, scalefile, maskfile, self.extents)
            np.save('grace{0}'.format(watershed), [self.twsData, self.nmonths, self.monthind, self.extents, self.lllon,self.lllat, self.cellsize])
        else:
            self.twsData, self.nmonths, self.monthind, self.extents,self.lllon,self.lllat, self.cellsize = np.load('grace{0}.npy'.format(watershed))
        #align the study period with GLDAS = 2002/04 to 2016/12               
        self.twsData = self.twsData[:-6,:] 
            
    def loadTWS(self, twsfile, scalefile, maskfile, extents):
        #form month index from 2002/4 to 2017/6
        def diff_month(d2, d1):
            return (d1.year - d2.year) * 12 + d1.month - d2.month +1
        
        nmonths = diff_month(datetime(2002,4,1), datetime(2017,6,1))
        print('number of grace months in the dataset=', nmonths)
        monthind = np.arange(nmonths)        
        #Remove the missing months (19 of them)
        #2002-06;2002-07;2003-06;2011-01;2011-06;2012-05;2012-10;2013-03;2013-08;
        #2013-09;2014-02;2014-12;2015-06;2015-10;2015-11;2016-04;2016-09;2016-10;
        #2017-02
        missingIndex = [2,3,14,105,110,121,126,131,136,137,142,148,152,158,162,163,168,173,174,178]
        monthind = np.delete(monthind, missingIndex)

        fh = Dataset(twsfile, mode='r')       
         
        self.lon = np.array(fh.variables['lon'][:])
        self.lat = np.array(fh.variables['lat'][:])
        self.times = np.array(fh.variables['time'][:])
        #convert times to actual time (not used?)
        t0 = date(2002, 1, 1)
        self.epoch = []
        for item in self.times:
            epochdate = t0 + timedelta(days=item)
            self.epoch.append(epochdate)

        gracevar = fh.variables['lwe_thickness']
        data = np.array(gracevar[:], dtype='float64')       
        fh.close()
        # 2004/1 to 2009/12 temporal mean already removed
        self.getMask(maskfile)
        self.getSF(scalefile)
        print('extents=', extents)
        ncells = (extents[1]-extents[0])*(extents[3]-extents[2])
        print(ncells)
        twsData = np.zeros((len(monthind), ncells))     
        
        for i in range(len(monthind)):
            dd = np.copy(data[i,:,:])
            #apply mask            
            dd = np.multiply(dd, self.land_mask)
            #apply scaling factor
            dd = np.multiply(dd, self.sf)                        
            #set null value
            #dd[dd>=3e+004]=np.nan
            DD = np.zeros(dd.shape, dtype='float64')            
            #shift 180
            DD[:, 360:720]=dd[:,0:360]
            DD[:,0:360] = dd[:,360:720]

            res = np.reshape(DD[extents[2]:extents[3], extents[0]:extents[1]], ncells)

            #in row-order (lon changes the fastest)
            twsData[i,:] = res
        #use a dataframe here to do linear interpolation
        newIndex = list(range(nmonths)) 
        df = pd.DataFrame(twsData, index= monthind)        
        df = df.reindex(newIndex)      
        df = df.interpolate(method='linear')
        #export as numpy array
        twsData = df.as_matrix()
        '''
        #for debugging        
        #plot tws data
        self.testmap = np.reshape(twsData[2,:], (extents[3]-extents[2], extents[1]-extents[0]))
        self.testmap[self.testmap==0] = np.NaN
        im = plt.imshow(self.testmap, cmap)
        cb = plt.colorbar(im)
        plt.savefig('gracetws.png')    
        '''
        return twsData, nmonths, monthind    
            
    def getMask(self, maskfile):        
        fh = Dataset(maskfile, mode='r')
        self.land_mask = np.array(fh.variables['land_mask'][:])        
        fh.close()

    def getSF(self, sffile):
        fh = Dataset(sffile, mode='r')
        self.sf = np.array(fh.variables['scale_factor'][:], dtype='float64')
        fh.close()
    
    def getExtent(self, bryfile):
        '''
        This gets study area boundary and form mask extent using GRACE grid info
        Coordinate information:
        geospatial_lat_min: -89.75
        geospatial_lat_max: 89.75
        geospatial_lat_units: degrees_north
        geospatial_lat_resolution: 0.5 degree grid
        geospatial_lon_min: 0.25
        geospatial_lon_max: 359.75        
        '''
        lllon=-180.0
        lllat=-90.0
        urlon=180.0
        urlat=90.0
        pixsize = 0.5
        
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
        #as03072018, had to subtract 0.5 for global area
        indx = np.ceil((locx-lllon-0.5)/pixsize)
        indy = np.ceil((locy-lllat)/pixsize)
        self.xind = indx
        self.yind = indy
        self.lllon = lllon
        self.lllat = lllat
        self.cellsize = pixsize
        return int(np.min(indx)), int(np.max(indx)), int(np.min(indy)), int(np.max(indy))


class NDVIData():
    '''
    Caution: the data are for India now
    '''
    def __init__(self, watershed, reLoad=False):
        if reLoad:
            bryfile = './bdy/%s.bdy' % (watershed)
            self.extents = self.getExtent(bryfile)
            self.ndviData = self.loadNDVI(self.extents)
            np.save('NDVI_%s.npy'%watershed, [self.ndviData])
        else:
            self.ndviData = np.load('NDVI_%s.npy'%watershed)
    
    def getExtent(self, bryfile):
        '''
        Data were subsetted from modis ndvi
        see ndviExtract
        The box is defined for India
        '''
        lllon=68.0
        lllat=6.75
        pixsize = 0.05
        
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
        indx = np.ceil((locx-lllon)/pixsize)
        indy = np.ceil((locy-lllat)/pixsize)
        self.xind = indx
        self.yind = indy
        return int(np.min(indx)), int(np.max(indx)), int(np.min(indy)), int(np.max(indy))

    def loadNDVI(self, extents, startYear=2002, endYear=2016):
        nmonths = (endYear-startYear+1)*12
        print('ndvi extents', extents)
        ndviData = np.zeros((nmonths, extents[3]-extents[2], extents[1]-extents[0]))
        counter=0
        scalefactor = 1e-4
        for iyear in range(startYear, endYear+1):
            for imon in range(1,13):
                data = np.load('./ndvi/{:4d}{:02d}.npy'.format(iyear,imon))
                data = data.reshape(-1, data.shape[-1])     
                data = data*scalefactor
                #subset
                ndviData[counter, :, :] = data[extents[2]:extents[3], extents[0]:extents[1]]
                '''
                plt.figure()
                im = plt.imshow(data, origin='lower')
                plt.colorbar(im)
                plt.savefig('ndvitest.png')
                '''
                counter+=1
        #return data corresponding to GRACE period
        #from 2002/04 to 2016/12
        return ndviData[3:,:,:]

class GLDASData():
    '''
    This is the main class for processing all data
    Get monthly tws simulated by gldas, data range 2000/1 to 2016/12 
    '''
    def __init__(self, watershed, startYear=2000, endYear=2016):
        '''
        watershed, name of the watershed file
        '''
        self.dataroot = '.'
        self.startYear = startYear
        self.endYear = endYear
        self.watershed = watershed
        self.gldastws = None
        self.inScaler = None
        self.outScaler = None
        self.needReset = True
        
    def loadStudyData(self, reloadData=False, masking=True):
        def diff_month(d2, d1):
            #returns elapsed months between two dates
            return (d1.year - d2.year) * 12 + d1.month - d2.month +1
        
        bryfile = '{0}/bdy/{1}.bdy'.format(self.dataroot, self.watershed)
        
        extents = self.getBasinPixels(bryfile)    
        print('basin extents is ', extents)
        self.mask = self.generateMask(bryfile)    
        print('mask size', self.mask.shape)
        
        #for debugging
        '''
        plt.figure()
        plt.imshow(self.mask)
        plt.savefig('nldasmask.png')
        '''
        if reloadData:  
            print('loading sm data ...')
            smMat = self.extractVar('sm200', extents, self.startYear,self.endYear)
            #form gldas_mask (this mask only removes the ocean cells)
            self.gldasmask = np.zeros((extents[3]-extents[2], extents[1]-extents[0]))+1.0
            dd = np.reshape(smMat[0,:], (self.gldasmask.shape))
            self.gldasmask[np.isnan(dd)]=0.0
            print(smMat.shape)
            print('loading canopy data ...')
            cpMat = self.extractVar('canopy', extents, self.startYear,self.endYear)
            print(cpMat.shape)
            print('loading snow water data ...')
            snMat = self.extractVar('snow', extents, self.startYear,self.endYear)
            print(snMat.shape)
            print('load precip data ...')
            pMat = self.extractVar('precip', extents, self.startYear, self.endYear)
            print(pMat.shape)
            
            #calculate tws
            tws = smMat+cpMat+snMat
            self.nldastwsB = tws            
            self.precipMat = pMat/10.0 #convert to cm
            #convert to cm to be consistent with grace data
            self.nldastwsB = self.nldastwsB/10.0
            #remove temporal mean from 2004/1 to 2009/12
            #to be consistent with grace tws processing
            meanmat = np.tile(np.mean(self.nldastwsB[(2004-2000)*12:(2009-2000+1)*12,:], axis=0), 
                              (self.nldastwsB.shape[0], 1))        

            self.nldastwsB = self.nldastwsB - meanmat
            np.save('{0}_twsdata'.format(self.watershed), [self.nldastwsB, self.precipMat, self.gldasmask])
        else:
            self.nldastwsB, self.precipMat, self.gldasmask = np.load('{0}_twsdata.npy'.format(self.watershed))
        self.extents = extents
        #
        #align nldas data to grace period
        #from 2002/04 to 2016/12 (note I need to use +3 because the zero-based index
        self.nldastws = self.nldastwsB[(2002-self.startYear)*12+3:,:]
        self.precipMat= self.precipMat[(2002-self.startYear)*12+3:,:]
        #extract cells inside mask
        if masking:
            #using basin mask
            self.validCells = np.where(self.mask==1)
            self.nvalidCells = len(self.validCells[0])
        else:
            #using simply the land mask from gldas
            if RESIZE==1:
                res = np.array(resize(self.gldasmask, output_shape=(N,N), preserve_range=True), dtype=np.int8)
            else:
                res = np.array(imresize(self.gldasmask, size=(N,N), mode='F', interp=INT_METHOD), dtype=np.int8)
            #extract valid cells
            self.validCells = np.where(res==1)
            self.nvalidCells = len(self.validCells[0])

        print('number of valid cells=%s' % self.nvalidCells)        
        print(self.nldastws.shape)
        print(self.precipMat.shape)
        
    def extractVar(self, varname, extents, startYear=2000, endYear=2016):
        '''
        @param varname, variable to be analyzed, keys defined in varnames
        @param extents [xmin, xmax, ymin, ymax]
        '''
        #loop through the files
        ncells = (extents[1]-extents[0])*(extents[3]-extents[2])
        d1 = date(2016, 12, 31)
        d2 = date(2000,1,1)

        nMonths = (d1.year - d2.year) * 12 + d1.month - d2.month +1        
       
        temp_folder = tempfile.mkdtemp()
        pmatfile = os.path.join(temp_folder, 'joblib_test.mmap')
        if os.path.exists(pmatfile): os.unlink(pmatfile)
        tempMat = np.memmap(pmatfile, dtype=np.double, shape=(nMonths, ncells), mode='w+')
        with Parallel(n_jobs=24) as parallelPool:
            parallelPool(delayed(getGLData)(tempMat, iyear, imon, 
                                            self.dataroot, varname, extents, ncells) 
                         for iyear in range(self.startYear, self.endYear+1) for imon in range (1,13))                        

        pMat = np.array(tempMat)
        return pMat
    
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
        '''
        This is the mask in CNN grid resolution and
        will be used extensively in the calculations if masking is set to True
        '''
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

    def formatInArray(self, arr, masking, varname='nldas'):
        '''
        Format input array for DL models 
        '''
        bigarr = None
        for i in range(arr.shape[0]):
            img0 = np.reshape(arr[i,:], (self.extents[3]-self.extents[2], 
                                self.extents[1]-self.extents[0]))        

            #img0[np.isnan(img0)] = 0 
            #02012018, I checked that the mass balance conserved after image resizing
            #print 'before', np.sum(img0)
            if RESIZE==1:
                res = np.array(resize(img0, output_shape=(N,N), preserve_range=True), dtype=np.float64)
            else:
                res = np.array(imresize(img0, size=(N,N), mode='F', interp=INT_METHOD), dtype=np.float64)
            
            #reset mask to exclude ocean areas. This is only done once
            if self.needReset:
                self.mask[np.isnan(res)]=0.0
                self.validCells = np.where(self.mask==1)
                self.nvalidCells = len(self.validCells[0])
                self.needReset=False
            
            if bigarr is None:
                bigarr = np.zeros((arr.shape[0], self.nvalidCells), dtype=np.float64)

            res[np.isnan(res)] = 0
            
            if masking:
                res=np.multiply(res,self.mask)

            bigarr[i,:] = res[self.validCells]
                
        if varname=='nldas':
            self.inScaler = skp.MinMaxScaler(feature_range=(XMIN, XMAX), copy=False)
            self.inScaler.fit_transform(bigarr)
        elif varname == 'precip':
            self.pScaler = skp.MinMaxScaler(feature_range=(XMIN, XMAX), copy=False)
            self.pScaler.fit_transform(bigarr)
        
        #values outside the mask will be zero, will this be right?
        resarr = np.zeros((bigarr.shape[0], N, N))
        for i in range(bigarr.shape[0]):
            temp = np.zeros((N,N))
            temp[self.validCells] = bigarr[i,:]
            resarr[i,:,:] = temp
            '''
            #for debugging normalized image
            plt.figure()
            im=plt.imshow(temp,cmap, origin='lower')
            plt.colorbar(im)   
            plt.savefig('gldasinput.png')
            sys.exit()
            '''
        return resarr
        
    def formatOutArray(self, arr1, arr2, extents2, masking):
        '''
        Format output array
        also calculates the spatially averaged tws series
        and the gldas-grace correlation matrix
        '''
        bigarr = np.zeros((arr1.shape[0], self.nvalidCells), dtype=np.float64)
        gldasArr = np.zeros((arr1.shape[0], self.nvalidCells), dtype=np.float64)
        self.graceArr = np.zeros((arr1.shape[0], self.nvalidCells), dtype=np.float64)
        for i in range(arr1.shape[0]):
            img0 = np.reshape(arr1[i,:], (self.extents[3]-self.extents[2], 
                                self.extents[1]-self.extents[0]))     
            img0[np.isnan(img0)] = 0.0         
            img1 = np.reshape(arr2[i,:], (extents2[3]-extents2[2],extents2[1]-extents2[0])) 
                            
            #normalizing data to the range [0, 1]
            #img0 /= maxval
            if RESIZE==1:
                d1 = np.array(resize(img0, output_shape=(N,N),preserve_range=True), dtype=np.float64)
                d2 = np.array(resize(img1, output_shape=(N,N),preserve_range=True), dtype=np.float64)
            else:
                d1 = np.array(imresize(img0, size=(N,N), mode='F', interp=INT_METHOD), dtype=np.float64)
                d2 = np.array(imresize(img1, size=(N,N), mode='F', interp=INT_METHOD), dtype=np.float64) 
            
            res = d1-d2                

            #for debugging, print out grace space average
            if i==10:               
                plt.figure(figsize=(12,6))
                plt.subplot(1,3,1)
                if masking:            
                    d1[np.where(self.mask==0)]=np.NaN
                im = plt.imshow(d1, cmap)
                plt.colorbar(im,  orientation="horizontal", fraction=0.046, pad=0.1)   
                plt.title('GLDAS/NOAH')
                
                plt.subplot(1,3,2)
                if masking:            
                    d2[np.where(self.mask==0)]=np.NaN
                im = plt.imshow(d2, cmap)
                plt.colorbar(im, orientation="horizontal", fraction=0.046, pad=0.1)   
                plt.title('GRACE')
                
                plt.subplot(1,3,3)
                if masking:
                    res[np.where(self.mask==0)]=np.NaN
                im = plt.imshow(res, cmap)
                plt.title('GLDAS-GRACE')
                plt.colorbar(im, orientation="horizontal", fraction=0.046, pad=0.1)
                plt.savefig('gldas-gracetest.png')

                
            if masking:
                res = np.multiply(res,self.mask)
                d1 = np.multiply(d1,self.mask)
                d2 = np.multiply(d2,self.mask)   
                     
            bigarr[i,:] = res[self.validCells]
            gldasArr[i,:] = d1[self.validCells]
            self.graceArr[i,:] = d2[self.validCells]
        
        gldas_grace_R = np.zeros((self.nvalidCells), dtype='float64')
        #calculate correlation between GLDAS and GRACE TWS at all grid pixels
        for i in range(self.nvalidCells):
            gldas_grace_R[i],_ = stats.pearsonr(gldasArr[:,i], self.graceArr[:,i])
        
        self.outScaler = skp.MinMaxScaler(feature_range=(XMIN, XMAX), copy=False)
        self.outScaler.fit_transform(bigarr)
        
        resarr = np.zeros((bigarr.shape[0], N, N))
        for i in range(bigarr.shape[0]):
            temp = np.zeros((N,N))
            temp[self.validCells] = bigarr[i,:]
            resarr[i,:,:] = temp
            '''
            #for debugging normalized image
            plt.figure()
            im=plt.imshow(temp,cmap)
            plt.colorbar(im)   
            plt.savefig('gldas-normalized.png')
            sys.exit()
            '''
        #convert correlation matrix to square
        self.gldas_grace_R = np.zeros((N,N))+np.NaN
        self.gldas_grace_R[self.validCells] = gldas_grace_R

        return resarr

    def formMatrix2DLSTM(self, gl, n_p=3, masking=False, nTrain=5000):
        '''
        Theano: (nsamples, times, channel, height, width)
        Tensorflow:  (nsamples, times, height, width, channel) <<used here>>
        gl = instance of GRACE (obtained using GRACE  class) 
        n_p, the number of antecedent states to use
        @param nmax, maximum lag
        '''
                  
        if self.nldastws is None:
            raise Exception('must loadstudyarea first')
        if gl is None:
            raise Exception('grace tws cannot be none')
        
        mat = self.nldastws        
        self.nmonths = mat.shape[0]
        maxval = np.amax(mat)
        gmat = gl.twsData
        gextents = gl.extents        
        #set up matrices for conv layers
        self.inArr = self.formatInArray(mat, masking)
        self.outArr = self.formatOutArray(mat, gmat, gextents, masking)

        if K.image_data_format()=="channels_first": #teano 
            Xtrain = np.zeros((nTrain-n_p, n_p, 1, N, N), dtype=np.float64)            
            Ytrain = np.zeros((nTrain-n_p, N, N), dtype=np.float64)
            
            for i in range(n_p, nTrain):
                Xtrain[i-n_p, :, 0, :, :] = self.inArr[i-n_p:i, :, :]                
                Ytrain[i-n_p, :, :] = self.outArr[i, :, :]
                
            Xtest = np.zeros((self.nmonths-nTrain, n_p, 1, N, N), dtype=np.float64)
            Ytest = np.zeros((self.nmonths-nTrain, N, N), dtype=np.float64)
            for i in range(nTrain, self.nmonths):
                Xtest[i-nTrain, :, 0, :, :] = self.inArr[i-n_p:i, :, :]
                Ytest[i-nTrain, :, :] = self.outArr[i, :, :]
        else: 
            Xtrain = np.zeros((nTrain-n_p, n_p, N, N, 1), dtype=np.float64)
            Ytrain = np.zeros((nTrain-n_p, N,N), dtype=np.float64)
            
            for i in range(n_p, nTrain):
                Xtrain[i-n_p, :, :, :, 0] = self.inArr[i-n_p:i, :, :]
                Ytrain[i-n_p, :,:] = self.outArr[i,:,:]
                
            Xtest = np.zeros((self.nmonths-nTrain, n_p, N, N, 1), dtype=np.float64)
            Ytest = np.zeros((self.nmonths-nTrain, N,N), dtype=np.float64)
            for i in range(nTrain, self.nmonths):
                Xtest[i-nTrain, :, :, :, 0] = self.inArr[i-n_p:i, :, :]
                Ytest[i-nTrain, :, :] = self.outArr[i,:,:]
            
        return Xtrain,Ytrain,Xtest,Ytest

    def getTWSAvg(self, mat, extents):
        '''
        calculate spatially averaged tws for both grace and nldas
        '''
        twsavg = np.zeros((mat.shape[0]))
        for i in range(mat.shape[0]):
            img = np.reshape(mat[i,:], (extents[3]-extents[2],extents[1]-extents[0])) 
            res = imresize(img, size=(N,N), mode='F', interp=INT_METHOD)
            res = np.multiply(res,self.mask)
            twsavg[i] = np.sum(res)/self.nvalidCells
        return twsavg
          
    def formMatrix2D(self, gl, n_p=3, masking=False, nTrain=1000):
        '''
        Form input and output data for DL models
        Theano: (nsamples, times, channel, height, width)
        Tensorflow:  (nsamples, times, height, width, channel) <<used here>>
        gl = instance of GRACE (obtained using GRACE  class) 
        n_p, the number of antecedent states to use
        @param nmax, maximum lag
        @param masking, if True then apply basin mask
        @param nTrain, samples used for training the DL 
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
        
        if self.watershed in ['india', 'indiabang']:
            print(self.inArr.shape)
            self.inArr[:, 9:21, 39] = 0.0 
            self.outArr[:,9:21, 39] = 0.0
            self.gldas_grace_R[9:21,39] = 0.0
            print(np.where(self.gldas_grace_R<0))
            
        # calculate the spatially averaged tws timeseries here
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

    def formPrecip2D(self, n_p=3, masking=False, nTrain=100):
        '''
        Form precipitation input data
        Theano: (nsamples, times, channel, height, width)
        Tensorflow:  (nsamples, times, height, width, channel) <<used here>>
        n_p, the number of antecedent states to use
        @param nmax, maximum lag
        @param masking, if True then apply basin mask
        @param nTrain, samples used for training the DL 
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

    def formNDVI2D(self, nv, n_p=3, masking=False, nTrain=106):
        '''
        Form NDVI input data
        I only implemented for tensorflow backend
        @param nv, instance of NDVIdata class
        @param n_p, max lag 
        @param masking, if True then apply basin mask
        @param nTrain, samples used for training the DL         
        '''            
        mat = nv.ndviData
        if len(mat.shape)==4: 
            mat = mat[0,:,:,:] 
        print('number of months=%s' % self.nmonths)
        print(mat.shape)
        #set up matrices for conv layers
        bigarr = np.zeros((mat.shape[0], self.nvalidCells), dtype=np.float64)
        for i in range(mat.shape[0]):
            img0 =  mat[i, :, :]
            #normalizing data to the range [0, 1]
            #img0 /= maxval
            if RESIZE==1:
                res = np.array(resize(img0, output_shape=(N,N),preserve_range=True), dtype=np.float64)
            else:
                res = np.array(imresize(img0, size=(N,N), mode='F', interp=INT_METHOD), dtype=np.float64)
            '''
            #for debugging, print out grace space average
            if i==0:               
                plt.figure(figsize=(12,6))
                plt.subplot(1,2,1)
                res[np.where(self.mask==0)]=np.NaN
                im = plt.imshow(res, cmap)
                plt.colorbar(im,  orientation="horizontal", fraction=0.046, pad=0.1)   
                plt.subplot(1,2,2)                
                im = plt.imshow(img0, cmap)
                plt.colorbar(im,  orientation="horizontal", fraction=0.046, pad=0.1)   
                plt.title('NDVI')                
                plt.savefig('ndvitest.png')
            '''
            if masking:
                res = np.multiply(res,self.mask)
                        
            bigarr[i,:] = res[self.validCells]

        outScaler = skp.MinMaxScaler(feature_range=(XMIN, XMAX), copy=False)
        outScaler.fit_transform(bigarr)
        resarr = np.zeros((bigarr.shape[0], N, N), dtype='float64')

        for i in range(bigarr.shape[0]):
            temp = np.zeros((N,N))
            temp[self.validCells] = bigarr[i,:]
            resarr[i,:,:] = temp
            '''
            #for debugging normalized image
            if i==10:
                plt.figure()
                im=plt.imshow(temp,cmap)
                plt.colorbar(im)   
                plt.savefig('ndvinormalized.png')
                sys.exit()
            '''
        Xtrain = np.zeros((nTrain-n_p+1, N, N, n_p), dtype=np.float64)
        
        for i in range(n_p-1, nTrain):
            for j in range(1,n_p+1):
                Xtrain[i-n_p+1, :, :, j-1] = resarr[i-n_p+j, :, :]
            
        Xtest = np.zeros((self.nmonths-nTrain+1, N, N, n_p), dtype=np.float64)
        
        for i in range(nTrain, self.nmonths):
            for j in range(1,n_p+1):
                Xtest[i-nTrain, :, :, j-1] = resarr[i-n_p+j, :, :]
                
        return Xtrain,Xtest

    def augData(self, X, masking=False):
        '''
        @X input array
        apply the same augmentation to input and output pairs?
        '''
        assert (len(X.shape)==4)
        Xg = np.zeros(X.shape, dtype='uint8')
        #convert input to range 0-255      
        for i in range(X.shape[0]):
            for j in range(X.shape[3]):
                Xg[i, :, :, j] = ((X[i, :, :, j] +1.0) * 255 / 2.0).astype('uint8')

        seq = iaa.Sequential([iaa.GaussianBlur((0, 1.5)), 
                              #iaa.Superpixels(p_replace=(0, 1.0),n_segments=(20, 200)),
                              iaa.AdditiveGaussianNoise(scale=0.05*255),
                              #iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)    
                              iaa.PiecewiseAffine(scale=(0.01, 0.05))])
  
        images_aug = seq.augment_images(Xg)
        #convert backc to (-1,1) range
        Xg = np.zeros(X.shape, dtype='float64')                
        for i in range(Xg.shape[0]):
            for j in range(Xg.shape[3]):
                Xg[i, :, :, j] = ((images_aug[i, :, :, j]*2.0)/255.0-1.0).astype('float64')
                if masking:
                    Xg[i, :, :, j] =np.multiply(Xg[i, :, :, j],self.mask)

        print(Xg.shape)                  
        '''        
        img = Xg[0, :, :, 0]
        plt.figure()
        plt.subplot(1,2,1)
        im = plt.imshow(img)
        plt.colorbar(im)
        plt.subplot(1,2,2)
        im = plt.imshow(X[0, :, :, 0])
        plt.colorbar(im)
        plt.savefig('imgaug.png')
        '''
        return Xg

    def getBasinBoundForPlotting(self):
        '''
        generate mask coordinates for plotting purposes
        '''
        bryfile = '{0}/bdy/{1}.bdy'.format(self.dataroot, self.watershed)
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
        xcellsize = float((self.extents[1]-self.extents[0]))/N
        ycellsize = float((self.extents[3]-self.extents[2]))/N
        #shift relative to the left-lower corder of study area
        print(xcellsize, ycellsize)
        maskX = (locx - (self.extents[0]*cellsize+xllcorner))/cellsize/xcellsize
        maskY = (locy - (self.extents[2]*cellsize+yllcorner))/cellsize/ycellsize
        
        return maskX, maskY
    
    def seasonalError(self, gl, masking=True):
        if self.nldastws is None:
            raise Exception('must loadstudyarea first')
        if gl is None:
            raise Exception('grace tws cannot be none')
        
        mat = self.nldastws        
        self.nmonths = mat.shape[0]
        gmat = gl.twsData
        gextents = gl.extents

        #calculate seasonal averages
        #extract data for 4 seasons, djf, mam, jja, son
        #set up matrices for conv layers
        inArr = self.formatInArray(mat, masking)
        outArr = self.formatOutArray(mat, gmat, gextents, masking)
        #period is from 2002/04 to 2016/12
        nMonth=inArr.shape[0]
        seasonind=[[[8], list(range(9,nMonth,12))], [[0, 1], list(range(12, nMonth, 12))],
                   [list(range(2, nMonth, 12))], [list(range(5, nMonth, 12))] 
                   ]
        meanErr = np.zeros((4,outArr.shape[1], outArr.shape[2]))
        labels=['DJF', 'MAM', 'JJF', 'SON']
        plt.figure()
        for i in range(4):
            ids = [x for item in seasonind[i] for x in item]
            print(ids)
            meanErr[i,:,:] = np.mean(outArr[ids,:,:], axis=0)
        
            plt.subplot(2,2,i+1)
            #im = plt.imshow(meanErr[i,:,:], cmap)
            #plt.colorbar(im, orientation="horizontal", fraction=0.046, pad=0.1)
            dd = meanErr[i,:,:]
            ax = sns.distplot(dd[self.validCells], label=labels[i])
            if i==0 or i==2:
                ax.set_ylabel('PDF')
            if i==2 or i==3:
                ax.set_xlabel('$TWSA_{GLDAS}-TWSA_{GRACE}$')
            plt.legend(loc='upper left')
        plt.savefig('meanerr.png')
        
        return meanErr
            
def main():
    '''
    valid watershed: India
    '''
    isMasking=True
    watershed='conus'
    print('start processing')
    grace = GRACEData(reLoad=True, watershed=watershed)
    gldas = GLDASData(watershed=watershed)
    gldas.loadStudyData(reloadData=True, masking=isMasking)

    Xtrain,Ytrain,Xtest,Ytest,Xval = gldas.formMatrix2D(gl=grace, n_p=3, masking=isMasking, nTrain=106)
    Ptrain, Ptest = gldas.formPrecip2D(n_p=3, masking=isMasking, nTrain=106)
    gldas.seasonalError(grace, masking=isMasking)
    

if __name__ == "__main__":
    main()