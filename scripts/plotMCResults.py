#Alex Sun
#date: 04082018
#purpose: plot monte carlo simulation results
#last update: 05132018
#==================================================================================
import matplotlib
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

def basinaverage(ax, modelname):

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.88, box.height])
    T0 = 2002+4.0/12
    counter=0
    nrz = 20
    allSeries = None
    for irz in range(-1, nrz):
        if irz==-1:
            model=modelname
            nTrain,rngTrain,rngTest,rngFull, arrTrain,graceTrain,arrTest,graceTest, noah = pkl.load(open('basincorr{0}.pkl'.format(model), 'rb'))
        else:
            model='{0}SRT'.format(modelname)
            nTrain,rngTrain,rngTest,rngFull, arrTrain,graceTrain,arrTest,graceTest, noah = pkl.load(open('basincorr{0}{1}.pkl'.format(model, irz), 'rb'))
            
        print(graceTrain.shape, graceTest.shape, rngFull.shape)

        tws = np.concatenate([graceTrain, graceTest])
        dlTWS = np.concatenate([arrTrain, arrTest])     
                      
        if allSeries is None:
            allSeries = dlTWS
        else:
            allSeries = np.c_[allSeries, dlTWS]
            
        if counter==0:
            ax.plot(rngFull, tws, '-o', markersize=4, color='#F39C12', linewidth=2.0, alpha=1.0, label='GRACE')
            ax.plot(rngFull, noah, ':', color='#626567', linewidth=2.0, alpha=1, label='NOAH')
                
        if counter==0:
            ax.axvspan(xmin=T0+(nTrain-0.1)/12, xmax=T0+(nTrain+0.1)/12, facecolor='#7B7D7D', linewidth=2.0)
        counter+=1
    ax.fill_between(rngFull, np.min(allSeries, axis=1), np.max(allSeries, axis=1), alpha=0.6)  
    ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.65), borderaxespad=0., fancybox=True, frameon=True)
    ax.set_xlabel('Year')
    ax.set_ylabel('TWSA(cm)')

if __name__ == "__main__":
    sns.set_style("ticks")        
    #modelname = 'ndvivgg'
    #modelname = 'compvgg'
    modelname = 'unetndvi'
    
    fig,axes=plt.subplots(1,1, figsize=(10,4.5), dpi=250)
    basinaverage(axes,modelname)
    plt.savefig('mc_results{0}.png'.format(modelname), dpi=fig.dpi)

    