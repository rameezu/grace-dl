#Alex Sun
#purpose: plot basin correlation traing/correction all together
##04292018, results are obtained by running convBasin.py
#05122018
#This generates Figure 4 of the paper (gldas_nsecorrplot)

import matplotlib
from boto.sns.connection import SNSConnection
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF

#set up font for plot
params = {'xtick.labelsize': 10,
          'ytick.labelsize': 10,
          'xtick.major.size': 1.5,      # major tick size in points
          'xtick.minor.size': .5,      # minor tick size in points
          'ytick.major.size': 1.5,      # major tick size in points
          'ytick.minor.size': .5,      # minor tick size in points
          'xtick.major.pad': 1,      # distance to major tick label
          'xtick.minor.pad': 1,
          'ytick.major.pad': 2,      # distance to major tick label
          'ytick.minor.pad': 2,
          'axes.labelsize': 10,
          'axes.linewidth': 1.0,
          'font.size': 12,
          'lines.markersize': 4,            # markersize, in points
          'legend.fontsize': 12,
          'legend.numpoints': 4,
          'legend.handlelength': 1.
          }
#matplotlib.rcParams.update(params)
import pickle as pkl
import seaborn as sns

models=['vgg16', 'compvgg', 'ndvivgg', 'unet', 'unetp', 'unetndvi']
#models=['vgg16', 'compvgg', 'ndvivgg']
labels = ['VGG16-1', 'VGG16-2', 'VGG16-3', 'Unet-1', 'Unet-2', 'Unet-3']

def basinaverage(ax):
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.88, box.height])
    T0 = 2002+4.0/12
    counter=0
    for item in models:
        print(item)
        nTrain,rngTrain,rngTest,rngFull, arrTrain,graceTrain,arrTest,graceTest, noah = pkl.load(open('basincorr{0}.pkl'.format(item), 'rb'))
        print ('loading {0}'.format('basincorr{0}.pkl'.format(item)))
        print (graceTrain.shape, graceTest.shape, rngFull.shape)

        if counter==0:
            ax.plot(rngFull, np.concatenate([graceTrain, graceTest]), '-o', markersize=4, color='#F39C12', linewidth=2.0, alpha=1.0, label='GRACE')
            ax.plot(rngFull, noah, ':', color='#626567', linewidth=2.0, alpha=1, label='NOAH')
        
        ax.plot(rngFull, np.concatenate([arrTrain, arrTest]), '-', label=labels[counter])

        if counter==0:
            ax.axvspan(xmin=T0+(nTrain-0.5)/12, xmax=T0+(nTrain+0.5)/12, facecolor='#7B7D7D', linewidth=2.0)
        counter+=1
        
    ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.45), borderaxespad=0., fancybox=True, frameon=True)
    ax.set_xlabel('Year')
    ax.set_ylabel('TWSA(cm)')
    ax.text(0.02,0.04, '(a)', fontsize=10, transform=ax.transAxes)


def cellaverageCorrelation(ax):
    '''
    This plots figure4b
    '''
    pltstyle=['-', '-', '-', '-', '-','-', '-']
    sns.set_style("ticks", {"xtick.major.size": 6, "ytick.major.size": 6})     
    counter=0    
    for modelname in models:
        corrvec, noahcorr = np.load('corrdist{0}.npy'.format(modelname))
        #remove bad cells
        corrvec = corrvec[np.where(np.isnan(corrvec)==False)[0]]
        cdf = ECDF(corrvec)    
        if counter==0:
            noahcorr = noahcorr[np.where(np.isnan(noahcorr)==False)[0]]
            noahcdf = ECDF(noahcorr, side='left')
            ax.plot(noahcdf.x, noahcdf.y, '--', color='#626567', label='NOAH', linewidth=1.0) 
        ax.plot(cdf.x, cdf.y, pltstyle[counter], label=labels[counter], linewidth=2.0)        
        
        counter+=1

    ax.set_xlabel('Correlation')
    ax.set_ylabel('CDF')
    ax.set_ylim(0, 1.0)
    ax.text(0.02,0.04, '(b)', fontsize=10, transform=ax.transAxes)
    ax.grid(True)
    ax.legend()

    #ax.text(0.02, 0.9, '(b)')

def cellaverageNSE(ax):
    '''
    This plots average NSE
    '''
    pltstyle=['-', '-', '-', '-', '-','-', '-']
    sns.set_style("ticks", {"xtick.major.size": 6, "ytick.major.size": 6})     
    counter=0    
    for modelname in models:
        corrvec, noahcorr = np.load('nsemat{0}.npy'.format(modelname))
        #remove bad cells
        corrvec = corrvec[np.where(np.isnan(corrvec)==False)[0]]
        cdf = ECDF(corrvec)    
        if counter==0:
            noahcorr = noahcorr[np.where(np.isnan(noahcorr)==False)[0]]
            noahcdf = ECDF(noahcorr, side='left')
            ax.plot(noahcdf.x, noahcdf.y, '--', color='#626567', label='NOAH', linewidth=1.0) 
        ax.plot(cdf.x, cdf.y, pltstyle[counter], label=labels[counter], linewidth=2.0)        
        
        counter+=1

    ax.set_xlabel('NSE')
    ax.set_ylabel('CDF')
    ax.set_ylim(0, 1.0)
    ax.set_xlim(-2, 1.0)
    ax.text(0.02,0.04, '(c)', fontsize=10, transform=ax.transAxes)
    ax.grid(True)



if __name__ == "__main__":
    sns.set_style("ticks")        

    fig,axes=plt.subplots(3,1, figsize=(12,10), dpi=300)
    basinaverage(axes[0])

    cellaverageCorrelation(axes[1])
    cellaverageNSE(axes[2])
    #plt.tight_layout(h_pad=0.05)
    plt.subplots_adjust(hspace=0.32)
    plt.savefig('gldas_nsecorrplot.eps', dpi=fig.dpi)