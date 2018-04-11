import pandas as pd
import numpy as np 
from datetime import datetime
import os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class ProcessIndia():
    def __init__(self, reLoad=False):
        dataRoot = './India'
        if reLoad:
            allfiles = os.listdir(dataRoot)
            self.dfAll = pd.DataFrame(columns=['wellid', 'date', 'waterlvl'])
            self.dfwellInfo = pd.DataFrame(columns=['wellid', 'state', 'district', 'wellname', 'welltype', 'lat', 'lon'])
        
            self.id = 0
            for afile in allfiles:
                afile = '/'.join([dataRoot, afile])
                print('processing,', afile)
                self.parseCSV(afile)
            print(self.dfwellInfo.shape)
            print(self.dfAll.shape)
            np.save('india.npy', [self.dfAll, self.dfwellInfo])
        else:
            self.dfAll, self.dfwellInfo = np.load('india.npy')
            
    def getAvg(self, State, startYear, endYear):
        df = self.dfwellInfo[self.dfwellInfo.state == State]
        
        #calculate mean of all wells
        allWells = df[['wellid', 'wellname']]
        #empty DF for holding goodf data
        cleanDF = pd.DataFrame(columns=['wellid', 'date', 'waterlvl'])
        for _, awell in allWells.iterrows():
            df0 = self.dfAll[self.dfAll['wellid']==awell['wellid']]
            #detect outliers
            u = df0['waterlvl']
            df0.iloc[:,2] = u - u.mean()             
            u = (u-u.mean())/u.std()
            #append good records to cleanDF
            cleanDF = cleanDF.append(df0[abs(u)<3.0], ignore_index = True)
        
        cleanDF = cleanDF[cleanDF.date.dt.year>=startYear]
        cleanDF = cleanDF[cleanDF.date.dt.year<=endYear]
        #now get monthly average * Sy* 100 [convert to cm]
        anomalyDF=cleanDF.groupby([cleanDF.date.dt.year, cleanDF.date.dt.month]).mean()*0.095*100.0
        
        return anomalyDF
        '''
        anomaly.plot(y='waterlvl','o-', style='-o', figsize=(12,5))
        plt.savefig('gwlevl.png')
            
        #print self.dfAll['waterlvl'].groupby(self.dfAll['wellid']).describe()

        df = self.dfAll.groupby(['wellid', 'wellname'])['wellid'].mean()
        print df
        for iyear in range(startYear, endYear+1):
            #get waterlvl values for the year
            pass
        ''' 
        
    def parseCSV(self, filename):
        '''
        January, April/May, August, and November
        '''
        df = pd.read_csv(filename, delimiter=',')
        df = df.sort_values(by=['WLCODE', 'YEAR_OBS'], ascending=True)
        #get a list of unique well names
        wellnames = pd.Index(df['WLCODE']).unique()
        for item in wellnames:
            print(item)
            #query all records for the current well
            welldf = df[df.WLCODE == item]            
            ADD=True
            tempDF = pd.DataFrame(columns=['wellid', 'date', 'waterlvl'])

            #first the well should have at least 12 rows            
            for _, row in welldf.iterrows():
                wellid = int(self.id)
                #for each year at least 3 should be available
                testval = np.array([float(row['MONSOON']), float(row['POSTMONSOONKHARIF']),
                                    float(row['POSTMONSOONRABI']), float(row['PREMONSOON'])])
                ind = len(np.where(np.isnan(testval))[0])
                if (ind>1):
                    #do not use this well?
                    ADD = False
                    break
                else:
                    d0 = [wellid, datetime(int(row['YEAR_OBS']), 1, 1), float(row['MONSOON'])]
                    d1 = [wellid, datetime(int(row['YEAR_OBS']), 4, 1), float(row['POSTMONSOONKHARIF'])]
                    d2 = [wellid, datetime(int(row['YEAR_OBS']), 8, 1), float(row['POSTMONSOONRABI'])]
                    d3 = [wellid, datetime(int(row['YEAR_OBS']), 11, 1), float(row['PREMONSOON'])]
                    df0 = pd.DataFrame([d0,d1,d2,d3], columns=['wellid', 'date', 'waterlvl'])
                    tempDF = tempDF.append(df0, ignore_index = True)
                
            if ADD:
                self.dfAll = self.dfAll.append(tempDF)
                self.dfwellInfo = self.dfwellInfo.append({'wellid':self.id,
                                                          'state':row['STATE'], 
                                                          'district':row['DISTRICT'], 
                                                          'wellname':row['WLCODE'], 
                                                          'welltype':row['SITE_TYPE'], 
                                                          'lat':row['LAT'], 
                                                          'lon':row['LON']}, ignore_index=True)

                self.id+=1
                print(self.id)
         
        return df
    
def main():
    india = ProcessIndia(reLoad=False)
    india.getAvg(State='PB', startYear=2002, endYear=2016)
if __name__ == "__main__":
    main()
    