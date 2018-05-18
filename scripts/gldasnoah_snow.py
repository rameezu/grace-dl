#===============================================
#Alex Sun
#Date: 20150208
#Updated on 20180216
#Download GLDAS NOAH
#This is for forcing data aggregation
#data source: GLDAS Noah Land Surface Model L4 monthly 0.25 X 0.25 degree V2.1
#change to parallel version
#===============================================

import urllib as url
import urlparse, gzip
import numpy as np
from numpy import loadtxt
import time

from subprocess import call
import datetime as dt
from datetime import timedelta,date
import calendar
import os
import sys
from joblib import Parallel, delayed, load, dump

def main():
    #
    xllcorner =-179.875
    yllcorner =-59.875
    cellsize =0.25
    nrows = 600
    ncols = 1440

    xmin = 0 
    xmax = ncols 
    ymin = 0 
    ymax = nrows 
    
    print 'lon %s, %s ' % (xmin, xmax)
    print 'lat %s, %s ' % (ymin, ymax)

    with Parallel(n_jobs=12) as parallelPool:
        #parallelPool(delayed(getNCFile)( iyear) for iyear in range(2000, 2017))
        parallelPool(delayed(getNCFile)( iyear) for iyear in range(2017, 2018))
        
def getNCFile(iyear):
    dataURL = r"https://hydro1.gesdisc.eosdis.nasa.gov/dods/GLDAS_NOAH025_M.2.1"
    varname = ['swe_inst'] #rain total

    d0 = (iyear-2000)*12
    for imon in range(1,13):
        filename = '/home/cc/gldas2/snow/snow%4d_%02d.nc'%(iyear, imon)
        #test if file already downloaded
        if not os.path.isfile(filename):         
            cmd = ["ncwa -O -v ", varname[0], ' -d time,%d,%d'%(d0,d0+1), ' -a time ', dataURL, ' ', filename]
            cmd =''.join(cmd)
            print 'executing %s' % cmd
            call(cmd, shell=True)
            d0+=1

main()
