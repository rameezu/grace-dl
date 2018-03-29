#alexsun: 03152018
#plot history of cnn training
import matplotlib
matplotlib.use('Agg')

import numpy as np
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
          'axes.labelsize': 9,
          'axes.linewidth': 1.0,
          'font.size': 12,
          'lines.markersize': 4,            # markersize, in points
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
        history = np.load('{0}history.npy'.format(modelname))
        history = history.tolist()
        loss =  history['rmse']
        print modelname, loss
        plt.plot(loss, pltstyle[counter], linewidth=1.5, label=labels[counter])        
        counter+=1
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.legend()
    plt.savefig('modelhistoryplt.png', dpi=fig.dpi)
        
if __name__ == "__main__":
    main()