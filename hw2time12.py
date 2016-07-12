import pandas as pd
import numpy as np
from pandas import Series, DataFrame

dataset = pd.read_csv("dataset.csv")

#print dataset.info()

##############################################
# Encoding categorical features
##############################################

dummies_Time_Period = pd.get_dummies(dataset['TIME_PERIOD'], prefix="Time_Period")
df = pd.concat([dataset, dummies_Time_Period], axis=1)
df.drop(['TIME_PERIOD','CRIME_DATE','TIME','CATDES','STATISTICAL_CODE_DESCRIPTION',], axis=1, inplace=True)
print df.info()

##############################################
# Scaling
##############################################

import sklearn.preprocessing as preprocessing
scaler = preprocessing.StandardScaler().fit(df)
#print scaler.mean_
#print scaler.scale_
scaler.transform(df)

##############################################
# Modeling
##############################################

#train_df = df.filter(regex='TIME_PERIOD|MONTH|DAY|YEAR|CATNO|ZIP|EVENT')
train_df = df
train_np = train_df.as_matrix()

# print train_df

y = train_np[:, 0]
X = train_np[:, 1:]

print "Sample Data:"
print train_np[0]
print train_np[1]


from sklearn import linear_model
clf = linear_model.LogisticRegression(C=1.0, penalty='l2', tol=1e-6, solver='newton-cg',multi_class='multinomial',class_weight='balanced')
clf.fit(X, y)
print "Classifier:"
print clf

import pickle
# now you can save it to a file
with open('whole_time12_balanced_lg.pkl', 'wb') as f:
    pickle.dump(clf, f)

# and later you can load it
# with open('filename.pkl', 'rb') as f:
#     clf = pickle.load(f)

##############################################
# Evaluation
##############################################

from sklearn import cross_validation
print cross_validation.cross_val_score(clf, X, y, cv=5)

#predicted = clf.predict(X)
#from sklearn import metrics
#print(metrics.classification_report(y, predicted))
#print(metrics.confusion_matrix(y, predicted))
