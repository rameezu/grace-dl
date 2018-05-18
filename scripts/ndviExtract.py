#===============================================
#Alex Sun
#Date: 20180210
#Download ndvi dataset from 
#https://e4ftl01.cr.usgs.gov/MOLT/MOD13C2.006/
#===============================================
import cmd
'''
import urllib as url
import urlparse, gzip
from numpy import loadtxt
import time
import json
from joblib import Parallel, delayed, load, dump
(90, -180.0, -90.0, 180.0)
shape (3600, 7200), resolution 0.05 global
'''
import numpy as np
from netCDF4 import Dataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os, sys
from subprocess import call
#extent of the dataset
xll = -180.0
yll = 90.0
nR = 3600
nC = 7200
#extent for the conUS
# xllcorner =-125.0
# yllcorner = 25.0
# cellsize =0.05
# nrows = 560
# ncols = 1160
#extent for big india
xllcorner =60.0
yllcorner =7.75
cellsize =0.05
nrows = 810
ncols = 810

def exactFile():
    startYear = 2017
    endYear = 2018
    col0 = np.int(((xllcorner-xll)/0.05))
    row0 = np.int(((yll-yllcorner)/0.05))

    for iyear in range(startYear,endYear):
        monthrng = range(1,13)
        for imon in monthrng:
            dataURL = r'https://e4ftl01.cr.usgs.gov/MOLT/MOD13C2.006/{:4d}.{:02d}.01/'.format(iyear,imon)
            cmd = 'wget -r -np {0} {1}'.format(dataURL, "-A '*.hdf'")
            print cmd
            call(cmd, shell=True)
            #now process the file
            ncdir=r'./e4ftl01.cr.usgs.gov/MOLT/MOD13C2.006/{:4d}.{:02d}.01/'.format(iyear,imon)
            ncfile = None
            for afile in os.listdir(ncdir):
                if afile.endswith(".hdf"):
                    ncfile = os.path.join(ncdir, afile)
                    break
            #extract the US part
            if not ncfile is None:
                with Dataset(ncfile, mode='r') as fh:  
                    var = fh.variables['CMG 0.05 Deg Monthly NDVI'][:]
                    data = var.data
                    ndvi = np.flipud(data[row0-nrows:row0, col0:col0+ncols])                    
                '''
                plt.figure()
                im = plt.imshow(ndvi, origin='lower')
                plt.colorbar(im)
                plt.savefig('testndvi.png')
                sys.exit()                  
                '''
                #save the US part
                np.save('./ndvi/{:4d}{:02d}.npy'.format(iyear,imon), [ndvi])
                #remove the downloaded file
                cmd = 'rm -Rf {0}'.format('./e4ftl01.cr.usgs.gov')
                print 'removing ', cmd
                call(cmd, shell=True)


def main():
    exactFile()

main()
