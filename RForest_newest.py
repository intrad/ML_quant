#coding=utf-8

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import datetime as dt

from sklearn import tree  
from sklearn.ensemble import RandomForestRegressor as DFR

# read in data
# here 0,5,10,...th days are selected, or this can be changed to 1,6,11,...th days by changing "first"
# path is the path of the csv file (string format)
# start is the first year of data we test
# end is the last year of data we test
def readin(first,path,start,end):
    temp = pd.read_csv(path,index_col=0)
    temp['date'] = temp.index
    temp = temp[temp['date'] >= start*10000]
    temp = temp[temp['date'] < (end+1)*10000]
    temp.pop('date')
    temp = temp.transpose().iloc[:,:]
    daynums = len(temp.columns)
    newdays = temp.columns[np.arange(first,daynums,5)]
    temp = temp[newdays]
    #print("tempocolumns:",temp.columns)
    #print("tempindex:",temp.index)
    return temp

def allreadin(path,start,end):
    temp = pd.read_csv(path)
    temp['date'] = temp.iloc[:,0]
    temp = temp[temp['date'] >= start*10000]
    temp = temp[temp['date'] < (end+1)*10000]
    temp.index = temp['date']
    temp = temp.transpose().iloc[1:-1,:]
    return temp


def getTrain(traindata,factorss,back_days,j,alldates,current_order,frt,num_factors):
	#get the data to be trained
    print("getting data of",alldates[current_order-back_days+1+j])
    mydata = pd.DataFrame(factorss[(current_order-back_days+1+j)%5][0][alldates[current_order-back_days+1+j]])
    mydata.columns = ['factor_0']
    for i in range(1,num_factors):
        to_add = pd.DataFrame(factorss[(current_order-back_days+1+j)%5][i][alldates[current_order-back_days+1+j]])
        to_add.columns = ['factor_'+str(i)]
        mydata = pd.concat([mydata,to_add],axis=1)
    mydata.index = factors[0].index 
    #print("mydata.index: ",mydata.index)

	#去除缺失值并取交集，得到的mydata为训练时的X，得到的target为训练时的y
    myreturn = pd.Series(frt[alldates[current_order-back_days+1+j]].loc[mydata.index])
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
#back_days is the length of training data, could be 1 or 5 or other number of days
num_factors = 40
back_days = 5
factor_path = '/Users/Momooo/Dropbox/transfer_cityu_data/sample 40/Factor data/'
return_path = '/Users/Momooo/Dropbox/transfer_cityu_data/backtest_data/5DForward.csv'
save_result_path = '/Users/Momooo/Dropbox/QT-JINYing/factors/factor.csv'
start_year = 2006
end_year = 2017
m_depth = 5

#read in the 50 factors as dataframes saved in "factors"
#the j_th element in factorss stores the factor data of j,j+5,j+10,...th day
factorss = []
for j in range(0,5):
    factors = []
    for i in range(0,num_factors):
        temp = readin(j,factor_path+str(i+1)+'.csv',start_year,end_year)
        #print("tempdates: ",temp.columns)
        factors.append(temp)
    factorss.append(factors)
print("####successfully read!",len(factorss),len(factorss[0]))
#print("factors0:",factorss[0][1],"factors1:",factorss[1][1],"factors2:",factorss[2][1])
stock_index = temp.index

#read in forward returns
#column is date, index is stock codes
frt = allreadin(return_path,start_year,end_year)
print("frt.index:",frt.index)
print("frt.columns:",frt.columns)
select_results = frt[:]

alldates = frt.columns
factor_results = pd.DataFrame(index=frt.index)
for j in range(0,5):
    order = j + 5
    current_result = None
    allreturns = pd.DataFrame(frt.columns,index = frt.columns)
    allreturns['return'] = 1
    validreturns = []
    #print(factor_results)
    
    while order < len(alldates):
    	#set current date to train decision tree
        current_date = alldates[order]
    
    	#get the data to be trained
    	#print(factorss[2])
    	#print(type(factorss),len(factorss))
    	#print("factor:",factorss[2][0])
        
        mydata = pd.DataFrame(factorss[j][0][current_date])
        print("####date:",current_date)
        mydata.columns = ['factor_0']
        for i in range(1,num_factors):
            #print("now dates: ",factorss[j][i].columns)
            to_add = pd.DataFrame(factorss[j][i][current_date])
            to_add.columns = ['factor_'+str(i)]
            mydata = pd.concat([mydata,to_add],axis=1)
        mydata.index = factors[0].index 
        #print("mydata index and columns:",mydata.index,mydata.columns)
        # make predictions if it is not the beginning day
        if order > 5 + j:
            mydata = mydata.dropna(axis=0)
            myindex = mydata.index
            mydata.index = myindex
    		
            '''
            predictions = current_result.predict(mydata)
            predictions = pd.DataFrame(predictions)
            predictions.index = mydata.index
            predictions.columns = ['result']
            positive = predictions[predictions['result']==1]
            negative = predictions[predictions['result']==-1]
            '''
            
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
        
        for t in range(0,back_days):
            traindata = getTrain(traindata,factorss,back_days,t,alldates,order,frt,num_factors)

        target = traindata.pop('target')
        print("traindata shape:",traindata.shape)

        #####set model and train
        dtr = DFR(n_estimators=50, max_depth=m_depth)
        current_result = dtr.fit(traindata,target)
        print("#training complete!")
    
        order = order + 5
    

'''
#get the list of everyday return 
allreturns.pop('date')
returns = allreturns.iloc[:,0].tolist()
print(returns)
IR = ((pd.Series(validreturns).mean() - 1)/pd.Series(validreturns).std() )* (252**0.5) / (5**0.5)
print("IR: ",IR)

#calculate the wealth curve
count = 0
money = [1]
while count < len(returns)-1:
    money.append(money[-1] * returns[count])
    count = count + 1

#print the wealth curve
results = pd.DataFrame(money)
results.columns = ['wealth']

def turn(date):
    date = str(date)
    return(dt.datetime.strptime(date,"%Y%m%d"))

results['date'] = alldates[:len(money)]
results['date'] = results['date'].apply(turn)

results.index = results['date']
results.plot(x='date',y='wealth',title='Wealth Curve',grid=True,legend=True)
plt.savefig('/Users/Momooo/Desktop/learn in THU/CityU RA/Quant相关/SVM/result_dist_linear_N=10_C=10.png')  
plt.show()

#save the stocks selected in csv file, where in each day, the stocks marked by "*" are selected
#print(type(select_results),select_results)
select_results.to_csv('select.csv')
'''
factor_results = factor_results.transpose()
factor_results = factor_results.sort_index()
print(factor_results.index)
print(factor_results.columns)
factor_results.to_csv(save_result_path)