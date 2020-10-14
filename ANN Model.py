# Importing the libraries
import numpy as np
import pandas as pd
import tensorflow as tf
#tf.__version__

file = pd.read_csv("Churn_Modelling.csv")
x = file.iloc[:,3:-1].values
y = file.iloc[:,-1].values

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
x[:,2] = le.fit_transform(x[:,2])

#for multiple categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
x = np.array(ct.fit_transform(x))

#for dummy values
x = x[:,1:]

from sklearn.model_selection import train_test_split 
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.25,random_state=0 )

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# importing the keras library and it's packges
import keras 
from keras.models import Sequential
from keras.layers import Dense

#making the ANN model 
classifier = Sequential(layers = None )

#adding the first layer and first hidden layer
classifier.add(Dense(output_dim = 6 , init = 'uniform',activation='relu',input_dim = 11))

# adding another hidden layer
classifier.add(Dense(output_dim = 6 , init = 'uniform',activation='relu'))

# adding output layer
classifier.add(Dense(output_dim = 1 , init = 'uniform',activation='sigmoid'))

#compiling the model
classifier.compile(optimizer='adam',loss = 'binary_crossentropy',metrics = ['accuracy'])

#fitting the model
classifier.fit(x_train, y_train,batch_size=10 ,epochs = 10)

y_predict = classifier.predict(x_test)
y_predict = (y_predict>0.5)

from sklearn.metrics import accuracy_score
ac=accuracy_score(y_test, y_predict)

                            