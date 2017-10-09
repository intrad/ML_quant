#coding=utf-8

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import datetime as dt
import string

# read in data
# here 0,5,10,...th days are selected, or this can be changed to 1,6,11,...th days by changing "first"
# path is the path of the csv file (string format)
# start is the first year of data we test
# end is the last year of data we test
def readin(first,path,start,end):
	temp = pd.read_csv(path)
	temp['date'] = temp.iloc[:,0]
	temp = temp[temp['date'] >= start*10000]
	temp = temp[temp['date'] < (end+1)*10000]
	temp.index = temp['date']
	temp = temp.transpose().iloc[1:-1,:]
	daynums = len(temp.columns)
	newdays = temp.columns[np.arange(first,daynums,5)]
	temp = temp[newdays]
	return temp

def allreadin(path,start,end):
	temp = pd.read_csv(path)
	temp['date'] = temp.iloc[:,0]
	temp = temp[temp['date'] >= start*10000]
	temp = temp[temp['date'] < (end+1)*10000]
	temp.index = temp['date']
	temp = temp.transpose().iloc[1:-1,:]
	return temp


def getTrain(traindata,factorss,back_days,j,alldates,dateorder,frt,num_factors):
	#get the data to be trained
	mydata = pd.DataFrame(factorss[j][0][alldates[dateorder-back_days+1+j]])
	mydata.columns = ['factor_0']
	for i in range(1,num_factors):
		to_add = pd.DataFrame(factorss[j][i][alldates[dateorder-back_days+1+j]])
		to_add.columns = ['factor_'+str(i)]
		mydata = pd.concat([mydata,to_add],axis=1)
	mydata.index = factors[0].index 
	#print("mydata.index: ",mydata.index)

	#去除缺失值并取交集，得到的mydata为训练时的X，得到的target为训练时的y
	myreturn = pd.Series(frt[alldates[dateorder-back_days+1+j]].loc[mydata.index])
	myreturn = myreturn.dropna(axis=0)
	mydata = mydata.loc[myreturn.index]

	myreturn = myreturn.sort_values(ascending=False)
	total = len(myreturn.index)
	#mark the first 10% stocks (by return) as 1, and the last 10% as -1
	top_index = myreturn.index[:int(total*0.1)]
	bottom_index = myreturn.index[int(total*0.9):]
	top = mydata.loc[top_index]
	bottom = mydata.loc[bottom_index]
	top['target'] = 1
	bottom['target'] = 0
	mydata = pd.concat([top,bottom],axis=0).dropna(axis=0)
	if traindata is None:
		return mydata
	else:
		return traindata.append(mydata)
		


##############################################
############### main program #################
##############################################

#####parameters
#num_factors is the number of the factors we want to test
#m_depth sets the maximum depth of decision tree
num_factors = 50
m_depth = 4
back_days = 5


from sklearn import tree  
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.ensemble import RandomForestClassifier as RFC

#read in the 50 factors as dataframes saved in "factors"
factorss = []
for j in range(0,back_days):
	factors = []
	for i in range(0,num_factors):
		temp = readin(j,'/Users/Momooo/Desktop/learn in THU/CityU RA/cityu_data/NiftyFiftyFactors/factor'+str(i+1)+'.csv',2008,2015)
		factors.append(temp)
		print("temp appended")
	factorss.append(factors)
print("####successfully read!",len(factorss),len(factorss[0]))
#print("factors0:",factorss[0][1],"factors1:",factorss[1][1],"factors2:",factorss[2][1])
stock_index = temp.index

#read in forward returns
#column is date, index is stock codes
frt = allreadin('/Users/Momooo/Desktop/learn in THU/CityU RA/cityu_data/ForwardReturn/5DForward.csv',2008,2015)
print("frt.index:",frt.index)
print("frt.columns:",frt.columns)
select_results = frt[:]


alldates = frt.columns
order = back_days - 1
current_result = None
allreturns = pd.DataFrame(frt.columns,index = frt.columns)
allreturns['return'] = 1
validreturns = []

while order < len(alldates):
	#set current date to train decision tree
	current_date = alldates[order]

	#get the data to be trained
	#print(factorss[2])
	#print(type(factorss),len(factorss))
	#print("factor:",factorss[2][0])
	mydata = pd.DataFrame(factorss[back_days-1][0][current_date])
	#print("####date:",mydata.columns)
	mydata.columns = ['factor_0']
	for i in range(1,num_factors):
		to_add = pd.DataFrame(factorss[back_days-1][i][current_date])
		to_add.columns = ['factor_'+str(i)]
		mydata = pd.concat([mydata,to_add],axis=1)
	mydata.index = factors[0].index 
	#print(mydata.index,mydata.columns)

	# if order>0 get the return of stocks selected by the decision tree trained before
	if order > back_days - 1:
		print("#####date:",current_date)
		mydata = mydata.dropna(axis=0)
		#print("mydata:",mydata.isnull().any().any())
		predictions = current_result.predict(mydata)

		predictions = pd.DataFrame(predictions)
		predictions.index = mydata.index
		predictions.columns = ['result']
		
		positive = predictions[predictions['result']==1]
		negative = predictions[predictions['result']==0]
		returns = frt[current_date]
		selected = returns.loc[positive.index]
		dropped = returns.loc[negative.index]

		positive_return = pd.Series(selected.tolist()).mean()
		negative_return = pd.Series(dropped.tolist()).mean()
		current_return = positive_return - negative_return
		if np.isnan(current_return):
			current_return = 0
		if current_return+1 < 0.8:
			print("$$$$$$$$$$$$$bullshit$$$$$$$$$$$")

		allreturns.ix[[current_date],'return'] = current_return + 1
		validreturns.append(current_return)
		print(current_return + 1)
		# mark the stocks selected by "*"
		select_results.ix[positive.index,[current_date]] = '*'


	#############################
	###train the decision tree###
#getTrain(traindata,factorss,j,alldates,dateorder,myreturn,num_factors):

	traindata = None
	for j in range(0,back_days):
		traindata = getTrain(traindata,factorss,back_days,j,alldates,order,frt,num_factors)
	#print("traindata:",traindata.isnull().any().any())
	#print(traindata,traindata.shape)
	target = traindata.pop('target')
	#print("########target:")
	#print(target,target.shape)
	#train the decision tree by gini(CART)
	dtc = RFC(criterion='gini',n_estimators=50,max_depth=m_depth)
	current_result = dtc.fit(traindata,target)

	order = order + 5

#get the list of everyday return 
allreturns.pop('date')
returns = allreturns.iloc[:,0].tolist()
print(returns)
IR = (pd.Series(validreturns).mean())/pd.Series(validreturns).std()
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
plt.show()

#save the stocks selected in csv file, where in each day, the stocks marked by "*" are selected
#print(type(select_results),select_results)
#select_results.to_csv('select.csv')

