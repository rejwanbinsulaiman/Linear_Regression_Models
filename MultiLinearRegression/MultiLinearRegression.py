# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 00:02:41 2021

Rizwan
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
dataset = pd.read_csv('50_Startups.csv') ## datasaet is available in the repo
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4]



# categorical veriables to numeric [encoding the catagorical variables]
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
from sklearn.compose import ColumnTransformer
lableencoder_X = LabelEncoder()
X[:,3] = lableencoder_X.fit_transform(X[:,3]) #[3 as the 4th column we need to encode on the X dataset]
ct = ColumnTransformer(
    [('0', # Just a name
      OneHotEncoder(), # The transformer class
      [3]) ## The column(s) to be applied on.
     ], 
    remainder='passthrough'
    )
X = np.array(ct.fit_transform(X), dtype=np.float)

#Avoiding the dummy variable trap
X = X[:, 1:]


#spliting the dataset into the training and test set
from sklearn.model_selection import train_test_split
X_train, X_test , y_train, y_test = train_test_split(X , y, test_size = 0.2 , random_state =0 )

#Fitting the model to the dataset
from sklearn.linear_model import LinearRegression
rg = LinearRegression()
rg.fit(X_train, y_train)

#Predicting in the test set
y_pred = rg.predict(X_test)

## Building the optimal model usig backward Elimination [Backward elimination prepration]
import statsmodels.api as sm  # instead of old statsmodel.formula.api we have to use statsmodels.api
X = np.append(arr = np.ones((50,1)).astype(int), values= X, axis =1)

# taking all the indexs off the columns on the X 
X_opt =X[:, [0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog= X_opt).fit()
print(regressor_OLS.summary())
# we removed index number 2 because the  P values is high
X_opt =X[:, [0,1,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog= X_opt).fit()
print(regressor_OLS.summary())
# we removed index number 1 because the  P values is high
X_opt =X[:, [0,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog= X_opt).fit()
print(regressor_OLS.summary())
# we removed index number 4 because the  P values is high
X_opt =X[:, [0,3,5]]
regressor_OLS = sm.OLS(endog = y, exog= X_opt).fit()
print(regressor_OLS.summary())
# we removed index number 5 because the  P values is high
X_opt =X[:, [0,3]] 
# as the number three index is there, it means  the the R&D spend,
# it means R&D is the most significant for the profitablity  
regressor_OLS = sm.OLS(endog = y, exog= X_opt).fit()
print(regressor_OLS.summary())
#This is not the perfect model, this is just to learn how to do backward elimination.
