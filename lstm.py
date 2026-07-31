import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM,Dense 
from keras.layers import Dropout
import matplotlib.dates as mdates


def initLSTM(train_X1, train_Y,test_X1, test_Y,scaler,testdate,cases):
    cases=cases.capitalize() 
    model = Sequential()
    model.add(LSTM(units=20,return_sequences=True,input_shape=(train_X1.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=40,return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=80,return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=80))
    model.add(Dropout(0.2))
    model.add(Dense(units=40,activation='relu'))
    model.add(Dense(units=40, activation='relu'))
    model.add(Dense(units=1, activation=None))
    model.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])
    model.fit(train_X1, train_Y, epochs=500, batch_size=10, validation_data=(test_X1, test_Y), shuffle=False)
    
    # make predictions
       
    testPredict = model.predict(test_X1)
    testPredict = scaler.inverse_transform(testPredict)
    test_Y1 = scaler.inverse_transform(test_Y)
    print("observedtestvalue: ",test_Y1)
    print("predictedtestvalue: ",testPredict)
    
    
    fig = plt.figure(4)
 #   fig.canvas.set_window_title('Observed Vs Predicted'+" "+cases+" "+'Cases in Testing Data Using LSTM')
    plt.title("Observed Vs Predicted"+" "+cases+" "+"Cases in Testing Data Using LSTM")  
    plt.xlabel("Date")  
    plt.ylabel(cases+" "+"Cases")
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator())
    plt.plot(testdate,test_Y1,'r-',)
    plt.plot(testdate,testPredict,'g-',)
    plt.gcf().autofmt_xdate()
    plt.legend(["Observed"+" "+cases+" "+"Cases in Test Data", "Predicted"+" "+cases+" "+"Cases Using LSTM in Test Data"])
    plt.show()
    
   