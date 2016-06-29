import pandas as pd
import numpy as np
from pandas import Series, DataFrame

dataset = pd.read_csv("dataset.csv")

print dataset.info()

train_df = dataset.filter(regex='TIME_PERIOD|MONTH|DAY|YEAR|CATNO|ZIP|EVENT')
train_np = train_df.as_matrix()

# print train_df

y = train_np[0:1000, 0]
X = train_np[0:1000, 1:]

print y
print X

from sklearn import linear_model
clf = linear_model.LogisticRegression(C=1.0, penalty='l2', tol=1e-6, solver='newton-cg',multi_class='multinomial')
clf.fit(X, y)

from sklearn import cross_validation
print cross_validation.cross_val_score(clf, X, y, cv=5)
