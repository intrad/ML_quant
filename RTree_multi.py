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
def readin(path,start,end):
	first = 0
	temp = pd.read_csv(path);
	temp['date'] = temp.iloc[:,0]
	temp = temp[temp['date'] >= start*10000]
	temp = temp[temp['date'] < (end+1)*10000]
	temp.index = temp['date']
	temp = temp.transpose().iloc[1:-1,:]
	daynums = len(temp.columns)
	newdays = temp.columns[np.arange(first,daynums,5)]
	temp = temp[newdays]
	return temp

##############################################
############### main program #################
##############################################

#####parameters
#num_factors is the number of the factors we want to test
#m_depth sets the maximum depth of decision tree
num_factors = 50
m_depth = 3
num_trees = 10

from sklearn import tree  
from sklearn.tree import DecisionTreeRegressor as DTR
from sklearn.cross_validation import train_test_split

#read in the 50 factors as dataframes saved in "factors"
factors = []
for i in range(0,num_factors):
    temp = readin('/Users/Momooo/Desktop/learn in THU/CityU RA/cityu_data/NiftyFiftyFactors/factor'+str(i+1)+'.csv',2011,2016)
    factors.append(temp)
print("####successfully read!")
stock_index = temp.index

#read in forward returns
#column is date, index is stock codes
frt = readin('/Users/Momooo/Desktop/learn in THU/CityU RA/cityu_data/ForwardReturn/5DForward.csv',2011,2016)
select_results = frt[:]

alldates = frt.columns
order = 0
current_result = None
allreturns = pd.DataFrame(frt.columns,index = frt.columns)
allreturns['return'] = 1

while order < len(alldates):
	#set current date to train d-tree
	current_date = alldates[order]

	#get the data to be trained
	mydata = pd.DataFrame(factors[0][current_date])
	mydata.columns = ['factor_0']
	for i in range(1,num_factors):
		to_add = pd.DataFrame(factors[i][current_date])
		to_add.columns = ['factor_'+str(i)]
		mydata = pd.concat([mydata,to_add],axis=1)
	mydata.index = factors[0].index 

	# if order>0 get the return of stocks selected by the decision tree trained before
	if order > 0:
		mydata = mydata.dropna(axis=0)
		predictions = current_result.predict(mydata)

		predictions = pd.DataFrame(predictions)
		predictions.index = mydata.index
		predictions.columns = ['result']

		pred_returns = predictions['result']
		pred_returns = pred_returns.sort_values(ascending=False)
		
		#get the index of long and short stocks
		positive_index = pred_returns.index[:int(len(pred_returns.index)/5)]
		negative_index = pred_returns.index[int(len(pred_returns.index)*4/5):]

		returns = frt[current_date]
		selected = returns.loc[positive_index]
		dropped = returns.loc[negative_index]
		positive_return = pd.Series(selected.tolist()).mean()
		negative_return = pd.Series(dropped.tolist()).mean()
		current_return = positive_return #- negative_return

		if np.isnan(current_return):
			current_return = 0
		allreturns.ix[[current_date],'return'] = current_return + 1

	#############################
	###train the decision tree###		

	#去除缺失值并取交集，得到的mydata为训练时的X，得到的target为训练时的y
	myreturn = pd.Series(frt[current_date].loc[mydata.index])
	myreturn = myreturn.dropna(axis=0)
	mydata = mydata.loc[myreturn.index]

	myreturn = myreturn.sort_values(ascending=False)
	total = len(myreturn.index)
	#mark the first 1/3 stocks (by return) as 1, and the last 1/3 as 0
	top_index = myreturn.index[:int(total/3)]
	bottom_index = myreturn.index[int(total*2/3):]
	top = mydata.loc[top_index]
	bottom = mydata.loc[bottom_index]
	top['target'] = myreturn.loc[top_index]
	bottom['target'] = myreturn.loc[bottom_index]
	mydata = pd.concat([top,bottom],axis=0).dropna(axis=0)
	target = mydata.pop('target')

	#train a new regression tree of today
	dtr = DTR(max_depth=m_depth)
	mytrees = []
	myscores = []
	for i in range(0,num_trees):
		train_X,test_X, train_y, test_y = train_test_split(mydata,target,test_size = 0.2,random_state = 11+2*i)
		newresult = dtr.fit(train_X,train_y)
		mytrees.append(newresult)
		y_predict = pd.Series(newresult.predict(test_X))
		test_y = pd.Series(test_y)

		test_y = test_y.sort_values(ascending=False)
		for rank in range(0,int(len(test_y)/2)):
			test_y[rank] = 1
		for rank in range(int(len(test_y)/2),len(test_y)):
			test_y[rank] = 0
		test_y = test_y.sort_index().tolist()

		y_predict = y_predict.sort_values(ascending=False)
		for rank in range(0,int(len(y_predict)/2)):
			y_predict[rank] = 1
		for rank in range(int(len(y_predict)/2),len(y_predict)):
			y_predict[rank] = 0
		y_predict = y_predict.sort_index().tolist()

		num_1 = y_predict.count(1)
		num_0 = y_predict.count(0)
		temp = test_y + y_predict
		num_11 = temp.count(2)
		num_00 = temp.count(0)
		newscore = (num_11/num_1)*(num_00/num_0)
		myscores.append(newscore)

	maxscore = max(myscores)
	current_result = mytrees[myscores.index(maxscore)]
	order = order + 1
	current_result = dtr.fit(mydata,target)

	order = order + 1

#get the list of everyday return 
allreturns.pop('date')
returns = allreturns.iloc[:,0].tolist()

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
print("####complete!")
plt.show()