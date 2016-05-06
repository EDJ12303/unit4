# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 21:41:01 2016

@author: Erin
"""

import numpy as np
import statsmodels.formula.api as smf
import pandas as pd
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt 
# Set seed for reproducible results
np.random.seed(414)

# Gen toy data
X = np.linspace(0, 15, 1000)
y = 3 * np.sin(X) + np.random.normal(1 + X, .2, 1000)

train_X, train_y = X[:700], y[:700]
test_X, test_y = X[700:], y[700:]

train_df = pd.DataFrame({'X': train_X, 'y': train_y})
test_df = pd.DataFrame({'X': test_X, 'y': test_y})

# Linear Fit
poly_1 = smf.ols(formula='y ~ 1 + X', data=train_df).fit()

# Quadratic Fit
poly_2 = smf.ols(formula='y ~ 1 + X + I(X**2)', data=train_df).fit()

#Linear model
ols_predict = poly_1.predict(train_df['X'])[:700]
#Quadratic model
quad_predict = poly_2.predict(train_df['X'])[:700]

#training set
print "ols_predict_training set"
plt.plot(ols_predict)

print "quad_predict_training set"
plt.plot(quad_predict)

## test set
ols_predict = poly_1.predict(test_df['X'])[700:]
#Quadratic model
quad_predict = poly_2.predict(test_df['X'])[700:]
print "ols_predict_test set"
plt.plot(ols_predict)
print "quad_predict_test set"
plt.plot(quad_predict)
