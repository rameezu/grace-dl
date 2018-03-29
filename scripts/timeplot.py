import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
def timeplot(train, test):
    '''
    Plot time series for each of the cells for each month
    Display training in green and test in blue
    '''
    
    n_train = len(twsTrain)
    train_idx = list(range(1,n_train+1))
    train_df = pd.DataFrame({'id':train_idx, 'val':twsTrain, 'type':np.repeat('train', n_train)})
    
    n_test = len(twsPred)
    test_idx = list(range(n_train+1, n_train+1+n_test))
    test_df = pd.DataFrame({'id':test_idx, 'val':twsPred, 'type':np.repeat('train', n_test)})
    
    full_df = pd.concat([train_df, test_df])
    
    
    full_df.plot(kind='line')
    
    plt.plot(train_df.id, train_df.val, '-', test_df.id, test_df.val, '--')
    
    
    
    train_idx = list(range(1,105))
    x_train = nldas.twsnldas[-X_train.shape[0]:]
    x_test = nldas.twsnldas[-X_test.shape[0]:]   
    
    test_idx = list(range(105,177))
    y_train = nldas.twsgrace[-Y_train.shape[0]:]
    y_test = nldas.twsgrace[-Y_test.shape[0]:]

    
    
    plt.plot(train_idx, x_train, '-', test_idx, x_test, '-')
    
    plt.plot(train_idx, y_train, '-.', test_idx, y_test, '-.')
    
    hist, bin_edges = np.histogram(X)
    plt.bar(bin_edges[:-1], hist, width = 1)
    plt.xlim(min(bin_edges), max(bin_edges))
    plt.show()   
    
    x = np.reshape(Y_test, (-1))
    n, bins, patches = plt.hist(x)
    plt.show()