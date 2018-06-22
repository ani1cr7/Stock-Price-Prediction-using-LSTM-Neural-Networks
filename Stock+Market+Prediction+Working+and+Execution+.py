
# coding: utf-8

# In[2]:

import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 
from pandas import datetime 
import math, time 
import itertools
from sklearn import preprocessing 
import datetime 
from operator import itemgetter 
from sklearn.metrics import mean_squared_error
from math import sqrt 
from keras.models import Sequential 
from keras.layers.core import Dense, Dropout, Activation 


# In[3]:

from keras.layers.recurrent import LSTM


# In[4]:

from keras.models import load_model 


# In[5]:

import keras 
import h5py
import requests 
import os 


# In[6]:

df = pd.read_csv("/Users/anirudhpanthula/Downloads/prices-split-adjusted.csv",index_col = 0)


# In[7]:

df.head() 


# In[8]:

df["adj close"] = df.close #Moving close to the last column  


# In[9]:

df.head()


# In[10]:

df.drop(['close'],1,inplace=True) 


# In[11]:

df.head()


# In[12]:

df2 = pd.read_csv("/Users/anirudhpanthula/Downloads/fundamentals.csv")


# In[13]:

df2.head()


# In[14]:

df2.columns


# # Extract all symbols from the list 

# In[15]:

symbols = list(set(df.symbol))


# In[16]:

len(symbols)


# In[17]:

symbols[:11] #Example of what is in the symbols 


# # Extract a particular price for stock in symbols 

# In[18]:

df = df[df.symbol == 'AAPL']
df.head()


# In[19]:

df.drop(['symbol'],1,inplace=True) #This is used to drop the column symbol 


# In[20]:

df.head()


# # Normalize the data 

# In[21]:

def normalize_data(df):
    min_max_scaler = preprocessing.MinMaxScaler() 
    df['open'] = min_max_scaler.fit_transform(df.open.values.reshape(-1,1))
    df['high'] = min_max_scaler.fit_transform(df.high.values.reshape(-1,1))
    df['low'] = min_max_scaler.fit_transform(df.low.values.reshape(-1,1))
    df['volume'] = min_max_scaler.fit_transform(df.volume.values.reshape(-1,1))
    df['adj close'] = min_max_scaler.fit_transform(df['adj close'].values.reshape(-1,1))
    return df 
df =normalize_data(df)
df.head() 


# # Create training set and testing set 

# In[22]:

def load_data(stock, seq_len):
    amount_of_features = len(stock.columns)
    data = stock.as_matrix()
    sequence_length = seq_len + 1 #index starting from 0 

    
    result = []
    
    for index in range(len(data) - sequence_length): #maximum date = latest date - sequence length 
        result.append(data[index: index + sequence_length]) #index: index +22days
    
    result = np.array(result)
    row = round(0.9 * result.shape[0]) #90% split
    train = result[:int(row), :]#all rows, all colummns here int is used in the bracket cus in the obove step row is multiplied by 0.9

    x_train = train[:, :-1] #all rows, all columns except last one 
    y_train = train[:, -1][:, -1]

    x_test = result[int(row):, :-1] #all rows, all but last one column
    y_test = result[int(row):, -1][:, -1]

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], amount_of_features))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], amount_of_features))

    return [x_train, y_train, x_test, y_test]

    
    


# important step above 
# data = stock.as_matrix() 
# Here the .as_matrix() is a pandas function, it converts any frame to its Numpy-array representation 
# dataFrame.as_matrix(columns=None)

# In[23]:

def load_data(stock, seq_len):
    amount_of_features = len(stock.columns)
    data = stock.as_matrix()
    sequence_length = seq_len + 1 #index starting from 0 

    
    result = []
    
    for index in range(len(data) - sequence_length): #maximum date = latest date - sequence length 
        result.append(data[index: index + sequence_length]) #index: index +22days
    
    result = np.array(result)
    row = round(0.9 * result.shape[0]) #90% split
    train = result[:int(row), :]#all rows, all colummns here int is used in the bracket cus in the obove step row is multiplied by 0.9

    x_train = train[:, :-1] #all rows, all columns except last one 
    y_train = train[:, -1][:, -1]

    x_test = result[int(row):, :-1] #all rows, all but last one column
    y_test = result[int(row):, -1][:, -1]

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], amount_of_features))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], amount_of_features))

    return [x_train, y_train, x_test, y_test]


# # Build the structure of the model 
# 

# Most suitable parameters 
# dropout = 0.3
# epochs = 90 
# 
# LSTM 256>LSTM 256> Relu 32>Linear 1 

# In[24]:

def build_model(layers):
    d = 0.3
    model = Sequential () 
    
    model.add(LSTM(256, input_shape=(layers[1], layers[0]), return_sequences=True))
    model.add(Dropout(d))
    model.add(LSTM(256, input_shape=(layers[1], layers[0]), return_sequences=False))
    model.add(Dropout(d)) 
    model.add(Dense(32,kernel_initializer='uniform', activation='relu'))
    model.add(Dense(1,kernel_initializer='uniform', activation = 'linear'))
    
    #adam = keras.optimizers.Adam(decay=0.2)
    
    start=time.time()
    model.compile(loss='mse',optimizer='adam', metrics=['accuracy'])
    print("compilation time: ", time.time()-start)
    return model 


# In[25]:

window = 22 
X_train, y_train, X_test, y_test = load_data(df, window)
print(X_train[0], y_train[0])


# In[26]:

model = build_model([5,window,1])


# In[27]:

model.fit(X_train,y_train,batch_size=512, epochs=90, validation_split=0.1,verbose=1)




# In[28]:

#print(X_test[-1])


# In[29]:

diff=[]
ratio=[]
p = model.predict(X_test)
print(p.shape)
#for each data index in test data 
for u in range(len(y_test)):
    #pr = predictation day u 
    pr = p[u][0]
    #(y_test day u / pr) - 1 
    ratio.append((y_test[u]/pr)-1)
    diff.append(abs(y_test[u]-pr))
    # print(u, y_test[u], pr, (y_test[u]/pr)-1, abs(y_test[u]- pr))
    # Last day prediction
    # print(p[-1]) 


# In[30]:

df = pd.read_csv("/Users/anirudhpanthula/Downloads/prices-split-adjusted.csv",index_col = 0)
df["adj close"] = df.close # Moving close to the last column
df.drop(['close'], 1, inplace=True) # Moving close to the last column
df = df[df.symbol == 'YHOO']
df.drop(['symbol'],1,inplace=True)

# Bug fixed at here, please update the denormalize function to this one
def denormalize(df, normalized_value): 
    df = df['adj close'].values.reshape(-1,1)
    normalized_value = normalized_value.reshape(-1,1)
    
    #return df.shape, p.shape
    min_max_scaler = preprocessing.MinMaxScaler()
    a = min_max_scaler.fit_transform(df)
    new = min_max_scaler.inverse_transform(normalized_value)
    return new

newp = denormalize(df, p)
newy_test = denormalize(df, y_test)


# In[31]:

def model_score(model, X_train, y_train, X_test, y_test):
    trainScore = model.evaluate(X_train, y_train, verbose=0)
    print('Train Score: %.5f MSE (%.2f RMSE)' % (trainScore[0], math.sqrt(trainScore[0])))

    testScore = model.evaluate(X_test, y_test, verbose=0)
    print('Test Score: %.5f MSE (%.2f RMSE)' % (testScore[0], math.sqrt(testScore[0])))
    return trainScore[0], testScore[0]


model_score(model, X_train, y_train, X_test, y_test)


# In[32]:

import matplotlib.pyplot as plt2

plt2.plot(newp,color='red', label='Prediction')
plt2.plot(newy_test,color='blue', label='Actual')
plt2.legend(loc='best')
plt2.show()


# In[ ]:



