# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 17:55:30 2016

@author: Erin
"""

import pandas as pd
loansData = pd.read_csv('https://spark-public.s3.amazonaws.com/dataanalysis/loansData.csv')
loansData['Interest.Rate'][0:5] 
loansData['Loan.Length'][0:5]
loansData['FICO.Range'][0:5]

#remove % from interest rate
cleanInterestRate = loansData['Interest.Rate'].map(lambda x: round(float(x.rstrip('%')) / 100, 4))
type(cleanInterestRate)

cleanInterestRate[0:5]

loansData['Interest.Rate'] = cleanInterestRate
loansData['Interest.Rate'][0:5]


# Loan.Length and FICO.Range
loansData['Loan.Length'][0:5]

cleanLoanLength = loansData['Loan.Length'].map(lambda x: int(x.rstrip(' months')))
cleanLoanLength[0:5]

loansData['Loan.Length'] = cleanLoanLength
loansData['Loan.Length'][0:5]

 # FICO.Range
loansData['FICO.Range'][0:5]


cleanFICORange = loansData['FICO.Range'].map(lambda x: x.split('-'))
cleanFICORange[0:5]

cleanFICORange[0:5].values[0]
['735', '739']
type(cleanFICORange[0:5].values[0])

cleanFICORange[0:5].values[0][0]

type(cleanFICORange[0:5].values[0][0])

cleanFICORange = cleanFICORange.map(lambda x: [int(n) for n in x])
cleanFICORange[0:5]

type(cleanFICORange[0:5].values[0])

loansData['FICO.Score'] = cleanFICORange.values[0][0]
loansData['FICO.Score'][0:5]
loansData['FICO.Score'][0:5]


a = pd.scatter_matrix(loansData, alpha=0.05, figsize=(10,10), diagonal='hist')
#build a model
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.cross_validation import KFold
from sklearn import linear_model, cross_validation, metrics
import math

intrate = loansData['Interest.Rate']
loanamt = loansData['Amount.Requested']
fico = loansData['FICO.Score']

# The dependent variable
y = np.matrix(intrate).transpose()
# The independent variables shaped as columns
x1 = np.matrix(fico).transpose()
x2 = np.matrix(loanamt).transpose()

#create input matrix
x = np.column_stack([x1,x2])

#create linear model
X = sm.add_constant(x)
model = sm.OLS(y,X)
f = model.fit()
# output results summary
print f.summary()

#k-fold 10 segments
train_X = X[:100]
test_X = X[100:]

train_y = y[:100]
test_y = y[100:]
kf = KFold(100, n_folds=10)
for train, test in kf:
    print(kf)
regr = linear_model.LinearRegression()
clf = regr.fit(train_X**4, train_y)
scores_mse = cross_validation.cross_val_score(clf, X, y, scoring = 'mean_squared_error' ,cv=10)
print (scores_mse)
print "mean MSE score="
print(scores_mse.mean())
scores_r2 = cross_validation.cross_val_score(clf, X, y, scoring = 'r2' ,cv=10)
print (scores_r2)
print "mean r2 score="
print (scores_r2.mean())
scores_mae = cross_validation.cross_val_score(clf, X, y, scoring = 'mean_absolute_error' ,cv=10)
print (scores_mae)
print "mean MAE score="
print (scores_mae.mean())