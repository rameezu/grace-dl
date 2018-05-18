#Alex Sun
#date: 04082018
#purpose: plot monte carlo simulation results
#Make plot for prediction period only
import matplotlib
import numpy as np
import datetime as dt
matplotlib.use('Agg')

import matplotlib.pyplot as plt

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
import numpy as np

PREDICT_DIR = 'pred'
def basinaverage(ax):

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.88, box.height])

    baseres, noah = pkl.load(open('{0}/base.pkl'.format(PREDICT_DIR), 'rb'))
    mcres = pkl.load(open('{0}/result.pkl'.format(PREDICT_DIR), 'rb'))
    meanres = np.mean(mcres, axis=0)
            
    basetws = baseres[-24:]
    meantws = meanres[-24:]
    mcres = mcres[:, -24:]
    noah = noah[-24:]
    #extract 2017 result only
    rng=[]
    for iy in [2016,2017]:
        for i in range(1,13):                        
            rng.append(dt.datetime(iy,i,1))

    #ax.plot(rng, basetws, '-', color='#2471A3', 
    #        linewidth=2.0, alpha=1.0, label='Base')
    ax.plot(rng, meantws, '-', linewidth=2.0, alpha=1.0, label='Ensemble mean')

    ax.plot(rng, noah, '--', linewidth=2.0, alpha=1.0, label='NOAH')
    
    ax.fill_between(rng, np.min(mcres, axis=0), np.max(mcres, axis=0), alpha=0.6, color='#BDC3C7', label='MC')  

    ax.set_xlabel('Time (month)')
    ax.set_ylabel('Predicted TWSA(cm)')
    ax.legend(loc='upper left')
    
if __name__ == "__main__":
    sns.set_style("ticks")        

    fig,axes=plt.subplots(1,1, figsize=(10,5), dpi=300)
    basinaverage(axes)
    plt.savefig('{0}/mc_results.eps'.format(PREDICT_DIR), dpi=fig.dpi)

    