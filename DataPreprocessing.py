# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 02:35:17 2021

@author: Rizwan
"""

import numpy as np
import pandas as pd

#import the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:,:-1].values # selecting all the rows and , :-1 means the except the last colums
y = dataset.iloc[:,3].values # 3 means only the last column
# add mean value in the missing dataset
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean' ) #using the  simputer
imputer = imputer.fit(X[:, 1:3]) # fiting the imputer in the x dataset
X[:, 1:3] =imputer.transform(X[:, 1:3]) #transforming the imputer in the x dataset

#Encoding the catagorical values
#we are using dataset X and only 0 index which is country and all the rows = X[:, 0] to encode from category to numbers

from sklearn.preprocessing import LabelEncoder , OneHotEncoder
from sklearn.compose import ColumnTransformer
lableencoder_X = LabelEncoder()
X[:,0] = lableencoder_X.fit_transform(X[:,0])

''' now we need to use colums of three dummy variables so the machine Learning do not catagory the countries based on 
number value

#     ColumnTransformer([('0', OneHotEncoder(), [0])], remainder='passthrough' )
'''
ct = ColumnTransformer(
    [('0', # Just a name
      OneHotEncoder(), # The transformer class
      [0]) ## The column(s) to be applied on.
     ], 
    remainder='passthrough'
    )
X = np.array(ct.fit_transform(X), dtype=np.float)

##encoding y label

lableencoder_y = LabelEncoder()
y = lableencoder_y.fit_transform(y)

#Spliting the data into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train , y_test = train_test_split(X,y, test_size = 0.2 , random_state= 0)

#feature scalling
# how much difference between the column and then making them balanced
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train) # we neeed to fit and transform on train set
X_test = sc_X.transform(X_test)# but for the test set we just transform the standardscaler object
## we do not need to scale dummy variable 
## but some time we lose the interpretation
# but we might get a very good acccuray
# scalling is important in the decicion tree as this make the algorith runs faster

