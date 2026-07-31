import pandas as pd   
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import lstm


 
#covid19cases='confirmed'
#covid19cases='recovered'
covid19cases='deceased'

date1='2021-01-01'
date2='2021-02-28'
districtname='Nagpur'

#=======================Code To Read Covid 19 India Datset=================================== 
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
######################parse_dates use to convert string type date into Date type
dataset=pd.read_csv("districts.csv", parse_dates=['Date'],dayfirst=True)
dataset = dataset[dataset['District'] != 'Unknown']

print("\n")
print("***********************Earlier Dates*****************************")
print("\n")
print(dataset.head()) 

#=======================Data Preprocessing====================================

dataset=dataset.drop(['State','Other','Tested'],axis=1)
dataset.columns=['date','district','confirmed','recovered','deceased']

print("\n")
print("***********************Current Dates*****************************")
print("\n")
print(dataset.tail())  

#============================Select Date====================================
#selectdate=dataset[dataset.date=='2020-05-01']
selectdate = dataset.loc[(dataset['date'] ==date1)]


#########Code to sort number of confirmed cases in descending order############
print("\n")
print("***********************Maximum Confirmed Cases*****************************")
print("\n")
maxconfirmedcases=selectdate.sort_values(by='confirmed',ascending=False)
print(maxconfirmedcases)

#Top 10 district of maximum confirmed cases
toptendistricts=maxconfirmedcases[0:7]
#print(toptendistricts)

# creating the bar plot
dictrict = toptendistricts['district'].tolist()
cases = toptendistricts['confirmed'].tolist()
  

#########Code to sort number of recovered cases in descending order############
print("\n")
print("***********************Maximum Recovered Cases*****************************")
print("\n")
maxrecoveredcases=selectdate.sort_values(by='recovered',ascending=False)
print(maxrecoveredcases)


print("\n")
print("***********************Maximum Deceased Cases*****************************")
print("\n")
maxdeceasedcases=selectdate.sort_values(by='deceased',ascending=False)
print(maxdeceasedcases)


###############Code For Prepare Data For LSTM and Bidirectional LSTM and CNN - LSTM model################################
selectdistrict=dataset[dataset['district'] == districtname]
# Filter data between two dates
selectdistrict = selectdistrict.loc[(selectdistrict['date'] >=date1) &
                                    (selectdistrict['date'] <= date2)]
selectdistrict=selectdistrict.sort_values(by='date')
#print(selectdistrict)
X=selectdistrict[['date']]
Y=selectdistrict[[covid19cases]]

#print("X: ",X)

train_X,test_X,train_Y,test_Y=train_test_split(X, Y, test_size=0.15,shuffle=False)

testdate=test_X
print("testdate: ",testdate)

#code to convert date into oordinal value
train_X['date']=train_X['date'].map(dt.datetime.toordinal)
test_X['date']=test_X['date'].map(dt.datetime.toordinal)



observedtestvalues=test_Y

#define min max scaler and transfor data
scaler = MinMaxScaler()
train_X=scaler.fit_transform(train_X)
test_X=scaler.fit_transform(test_X)
train_Y=scaler.fit_transform(train_Y)
test_Y=scaler.fit_transform(test_Y)

#print("Normalize test_Y: ",test_Y)

print("train_X.shape[0]",train_X.shape[0])
print("train_X.shape[1]",train_X.shape[1])

# reshape input to be 3D for LSTM and Bidirectional [samples, timesteps, features]
train_X1 = train_X.reshape((train_X.shape[0], 1, 1))
test_X1= test_X.reshape((test_X.shape[0], 1, 1))
print(train_X.shape, train_Y.shape, test_X.shape, test_Y.shape)

##function calling LSTM
lstm.initLSTM(train_X1, train_Y,test_X1, test_Y,scaler,testdate,covid19cases)


