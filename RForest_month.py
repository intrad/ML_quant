#coding=utf-8

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import datetime as dt
import copy

from sklearn import tree  
from sklearn.ensemble import RandomForestRegressor as DFR

# read in data
# the index of the output is the index of stocks, the columns of the output are dates
# path is the path of the csv file (string format)
# start is the first year of data we test
# end is the last year of data we test
def readin(path,start,end):
    temp = pd.read_csv(path,index_col=0)
    temp['date'] = temp.index
    temp['date'] = temp['date'].apply(lambda x:int(x[0:4]+x[5:7]+x[8:10]))
    temp.index = temp['date']
    temp = temp[temp['date'] >= start*10000]
    temp = temp[temp['date'] < (end+1)*10000]
    temp.pop('date')
    temp = temp.transpose().iloc[:,:]
    return temp


def getTrain(traindata,factors,frt,date,num_factors):
	#get the data to be trained
    print("getting data of",date)
    mydata = pd.DataFrame(factors[0][date])
    mydata.columns = ['factor_0']
    for i in range(1,num_factors):
        to_add = pd.DataFrame(factors[i][date])
        to_add.columns = ['factor_'+str(i)]
        mydata = pd.concat([mydata,to_add],axis=1)
    mydata.index = factors[0].index 
    #print("mydata.index: ",mydata.index)

	#去除缺失值并取交集，得到的mydata为训练时的X，得到的target为训练时的y
    myreturn = pd.Series(frt[date].loc[mydata.index])
    myreturn = myreturn.dropna(axis=0)
    mydata = mydata.loc[myreturn.index]

    #按收益排序
    myreturn = myreturn.sort_values(ascending=False)
    total = len(myreturn.index)
	#mark the first 10% stocks (by return) as 1, and the last 10% as -1
    top_index = myreturn.index[:int(total*0.1)]
    bottom_index = myreturn.index[int(total*0.9):]
	#print("num of top: ",len(top_index),"num of bottom: ",len(bottom_index))
    top = mydata.loc[top_index]
    bottom = mydata.loc[bottom_index]
    top['target'] = 1
    bottom['target'] = -1
    mydata = pd.concat([top,bottom],axis=0).dropna(axis=0)
	#print("num of stocks: ",len(mydata.index))
    if traindata is None:
        return mydata
    else:
        return traindata.append(mydata)
		


##############################################
############### main program #################
##############################################

#####parameters#####
#num_factors is the number of the factors we want to test
#back_months is the length of training data, could be 1 or 5 or other number of months
num_factors = 40
back_months = 5
factor_path = '/Volumes/Seagate Backup Plus Drive/Dropbox/data_cn/factor/'
return_path = '/Volumes/Seagate Backup Plus Drive/Dropbox/data_cn/FWRTN1M.csv'
save_result_path = '/Volumes/Seagate Backup Plus Drive/Dropbox/data_cn/bt_result/Rforest_factor.csv'
start_year = 2006
end_year = 2017
m_depth = 5

#the j_th element in factors stores the ith factor data
factors = []
for j in range(0,num_factors):
    temp = readin(factor_path+str(j+1)+'.csv',start_year,end_year)
    #print("tempdates: ",temp.columns)
    factors.append(temp)
print("####successfully read!",len(factors))
#print("factors0:",factorss[0][1],"factors1:",factorss[1][1],"factors2:",factorss[2][1])


#read in forward returns
#column is date, index is stock codes
frt = readin(return_path,start_year,end_year)
print("frt.index:",frt.index)
print("frt.columns:",frt.columns)
select_results = copy.copy(frt)

#所有日期
alldates = frt.columns
factor_results = pd.DataFrame(index=frt.index)

order = back_months
current_result = None
allreturns = pd.DataFrame(frt.columns,index = frt.columns)
allreturns['return'] = 1
validreturns = []
#print(factor_results)
    
while order < len(alldates):
#set current date to train
    current_date = alldates[order]
        
    mydata = pd.DataFrame(factors[0][current_date])
    print("####date:",current_date)
    mydata.columns = ['factor_0']
    for i in range(1,num_factors):
        to_add = pd.DataFrame(factors[i][current_date])
        to_add.columns = ['factor_'+str(i)]
        mydata = pd.concat([mydata,to_add],axis=1)
    mydata.index = factors[0].index 
    #print("mydata index and columns:",mydata.index,mydata.columns)
    
    # make predictions if it is not the beginning day
    if order > back_months:
        mydata = mydata.dropna(axis=0)
        myindex = mydata.index
#        mydata = pd.DataFrame(preprocessing.scale(mydata))
        mydata.index = myindex
            
        # distance calculates the signed distance from the sample to the separation hyperplane
        # the factor value is exactly the signed distance
        tempresult = current_result.predict(mydata)
        tempresult = pd.DataFrame(tempresult)
        tempresult.columns = [current_date]
        tempresult.index = myindex
        #print("####distance: ",distance)
        #save the factor value
        factor_results = pd.concat([factor_results,tempresult],axis=1)
        print("#result shape:::" ,factor_results.shape)

    ######################################
    ###train new model for 5 days later###

    traindata = None
    #getTrain(traindata,factors,frt,date,num_factors):
        
    for t in range(0,back_months):
        traindata = getTrain(traindata,factors,frt,alldates[order-t],num_factors)

    target = traindata.pop('target')
    #print("traindata shape:",traindata.shape)
    #####standardize
#    traindata = pd.DataFrame(preprocessing.scale(traindata))
    #####set model and train
    dtr = DFR(n_estimators=50, max_depth=m_depth)
    current_result = dtr.fit(traindata,target)
    #print("#training complete!")
    
    order = order + 1
    

factor_results = factor_results.transpose()
factor_results = factor_results.sort_index()
print(factor_results.index)
print(factor_results.columns)
factor_results.to_csv(save_result_path)


