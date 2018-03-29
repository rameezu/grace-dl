#Alex Sun
#purpose: plot basin correlation traing/correction all together
#Figure 4 of the paper
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
matplotlib.rcParams.update(params)
import pickle as pkl
import seaborn as sns

def basinaverage(ax):
    models=['simple', 'vgg16', 'comp', 'ndvicnn']
    labels = ['CNN-NOAH', 'VGG16', 'CNN-Comp1', 'CNN-Comp2']

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.88, box.height])
    T0 = 2002+4.0/12
    counter=0
    for item in models:
        nTrain,rngTrain,rngTest,rngFull, arrTrain,graceTrain,arrTest,graceTest, noah = pkl.load(open('basincorr{0}.pkl'.format(item)))
        print graceTrain.shape, graceTest.shape, rngFull.shape

        if counter==0:
            ax.plot(rngFull, np.concatenate([graceTrain, graceTest]), '-o', markersize=4, color='#F39C12', linewidth=2.0, alpha=1.0, label='GRACE')
            ax.plot(rngFull, noah, ':', color='#626567', linewidth=2.0, alpha=1, label='NOAH')
        
        ax.plot(rngFull, np.concatenate([arrTrain, arrTest]), '-', label=labels[counter])

        if counter==0:
            ax.axvspan(xmin=T0+(nTrain-0.5)/12, xmax=T0+(nTrain+0.5)/12, facecolor='#7B7D7D', linewidth=2.0)
        counter+=1
        
    ax.grid(False)
    ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.65), borderaxespad=0., fancybox=True, frameon=True)
    ax.set_xlabel('Year')
    ax.set_ylabel('TWSA(cm)')
    #ax.text(0.02, 0.90, '(a)')    

def cellaverage(ax):
    models=['simple', 'vgg16', 'comp', 'ndvicnn']
    labels = ['CNN-NOAH', 'VGG16', 'CNN-Comp1', 'CNN-Comp2']
    pltstyle=['-', '--', ':', '-.']
    
    counter=0    
    for modelname in models:
        corrvec, noahcorr = np.load('corrdist{0}.npy'.format(modelname))
        #remove bad cells
        corrvec = corrvec[np.where(np.isnan(corrvec)==False)[0]]
        cdf = ECDF(corrvec)    
        if counter==0:
            noahcorr = noahcorr[np.where(np.isnan(noahcorr)==False)[0]]
            noahcdf = ECDF(noahcorr, side='left')
            ax.plot(noahcdf.x, noahcdf.y, '-', color='#626567', label='NOAH', linewidth=1.0) 
        ax.plot(cdf.x, cdf.y, pltstyle[counter], label=labels[counter], linewidth=2.0)        
    
        counter+=1

    ax.set_xlabel('CNN/GRACE Correlation')
    ax.set_ylabel('CDF')
    ax.set_ylim(0, 1.0)
    ax.grid(True)
    ax.legend()

    #ax.text(0.02, 0.9, '(b)')

if __name__ == "__main__":
    sns.set_style("white")    
    fig,axes=plt.subplots(1,1, figsize=(10,4), dpi=250)
    basinaverage(axes)
    plt.savefig('gldas_timeseriesAll.eps', dpi=fig.dpi)

    fig,axes=plt.subplots(1,1, figsize=(8,4), dpi=250)
    cellaverage(axes)
    plt.savefig('gldas_cellcorrelationdist.eps', dpi=fig.dpi)
    