# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 20:00:31 2022

@author: megha
"""

# Import libraries
import numpy as np
import pandas as pd

#read the data
df = pd.read_csv(r'C:\Users\megha\OneDrive\Documents\DockerOnEC2Files\online_sales.csv')

df.shape

df.head()

#target class frequency
df.converted.value_counts()

df.info()

df.describe()

input_columns = [column for column in df.columns if column != 'converted']
output_column = 'converted'
print (input_columns)
print (output_column)

#input data
X = df.loc[:,input_columns].values
#output data 
y = df.loc[:,output_column]
#shape of input and output dataset
print (X.shape, y.shape)

#import model specific libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

#Split the data into training and test data (70/30 ratio)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3, random_state=555, stratify=y)

#validate the shape of train and test dataset
print (X_train.shape)
print (y_train.shape)

print (X_test.shape)
print (y_test.shape)

#check on number of positive classes in train and test data set
print(np.sum(y_train))
print(np.sum(y_test))

#fit the logisitc regression model on training dataset 
logreg = LogisticRegression(class_weight='balanced').fit(X_train,y_train)

logreg.score(X_train, y_train)

#validate the model performance on unseen data
logreg.score(X_test, y_test)

#make predictions on unseen data
predictions=logreg.predict(X_test)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

print(classification_report(y_test, predictions,target_names=["Non Converted", "Converted"]))

logreg

### Create a Pickle file using serialization 
import pickle
pickle_out = open("logreg.pkl","wb")
pickle.dump(logreg, pickle_out)
pickle_out.close()

pickle_in = open("logreg.pkl","rb")
model=pickle.load(pickle_in)

model

#predict using the model on customer input
model.predict([[32,1,1]])[0]

df_test=pd.read_csv('test_data.csv')
predictions=model.predict(df_test)

print(list(predictions))
