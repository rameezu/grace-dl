#Alex Sun
#Date: 03152018
#purpose:plotcorrelation distribution from all models
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
          'axes.labelsize': 9,
          'axes.linewidth': 1.0,
          'font.size': 12,
          'lines.markersize': 2,            # markersize, in points
          'legend.fontsize': 10,
          'legend.numpoints': 2,
          'legend.handlelength': 1.
          }
matplotlib.rcParams.update(params)

def main():
    models=['simple', 'vgg16', 'comp', 'ndvicnn']
    labels = ['NOAH only', 'VGG16', 'NOAH+P', 'NOAH+P+NDVI']
    pltstyle=['-', '--', ':', '-.']
    
    fig=plt.figure(figsize=(8,4), dpi=250)
    counter=0    
    for modelname in models:
        corrvec = np.load('corrdist{0}.npy'.format(modelname))
        cdf = ECDF(corrvec)        
        plt.plot(cdf.x, cdf.y, pltstyle[counter], label=labels[counter], linewidth=1.5)        
        counter+=1
    plt.xlabel('CNN/GRACE Correlation')
    plt.ylabel('CDF')
    plt.legend()
    plt.savefig('corrcdfplt.png', dpi=fig.dpi)
        
if __name__ == "__main__":
    main()
     