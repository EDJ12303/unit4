# -*- coding: utf-8 -*-
"""
Created on Sun May 22 16:42:31 2016

@author: Erin
"""

import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.naive_bayes import GaussianNB
import sklearn.metrics as skm

#input csv
df = pd.read_csv('C:/Users/Erin/thinkful/Unit4/ideal_weight.csv')


#rename columns without quotations marks
cols = df.columns
cols = cols.map(lambda x: x.replace("'",' ') if isinstance(x, (str, unicode)) else x)
df.columns = cols

#strip whitespace
df.rename(columns=lambda x: x.strip(), inplace=True)
#check columns
print df[0:10]
print df['sex'][0:10]


#remove quotes from df['sex'] values
cols = df['sex']
cols = cols.map(lambda x: x.replace("'",' ') if isinstance(x, (str, unicode)) else x)
df['sex'] = cols
print df['sex'][0:10]

#plot distributions of actual and ideal weight
plt.figure()
df['actual'].plot.hist(orientation='vertical', cumulative=False, bins=20)
df['ideal'].plot.hist(orientation='vertical', cumulative=False, bins=20)

#plot distribution of difference in actual/ideal weight
plt.figure()
df['diff'].plot.hist(orientation='vertical', cumulative=False, bins=20)

#categorical variable for 'sex'
df['sex'] = df['sex'].astype('category')

#more women than men?
df.groupby(['sex']).size()
print df.groupby(['sex']).size()
#females = 119
#males = 63

#Fit classifier 'sex' to actual weight, ideal weight, and diff
clf = GaussianNB()
Y = df.sex
Y = np.array(Y)
X = df[['actual','ideal','diff']]
X = np.array(X)
clf.fit(X,Y)

#Predictions
y_pred = clf.fit(X,Y).predict(X)

#How many points were mislabeled?
print('Mislabeled points out of a total of %d points : %d' % (X.shape[0], (Y != y_pred).sum()))

#Predict the sex for an actual weight of 145, an ideal weight of 160, and a diff of -15
print clf.predict([[145, 160, -15]])
#['Male']

#Predict the sex for an actual weight of 160, an ideal weight of 145, and a diff of 15
print clf.predict([[160, 145, 15]])
#['Female']





 


  
    







