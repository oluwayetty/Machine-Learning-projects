"""
    Let's build our model :)
"""
import pandas as pd
import numpy as np
import pdb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression as LG
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.metrics import precision_score as precision
from sklearn.metrics import recall_score as recall
from sklearn.metrics import f1_score as f1
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

from imblearn.over_sampling import SMOTE, ADASYN

datasets = pd.read_csv("./processed_data/datasets.csv")

# encoding our class value [True =1. False = 0]
label_encoder = LabelEncoder()
class_integer_encoded  = label_encoder.fit_transform(datasets['Class'])

X = datasets.iloc[:,2:10]
y = class_integer_encoded

# resampling with SMOTE
X_resampled, y_resampled = SMOTE().fit_resample(X, y)

# splitting my data in ratio of 30:70 percent
features_train, features_test, labels_train, labels_test = train_test_split(X_resampled, y_resampled, stratify = y_resampled, test_size=0.3)

# train our model
clf = RFC().fit(features_train, labels_train)

# use our model to predict
predictions_class_test = clf.predict(features_test)

#compute accuracy_score
accuracy = metrics.accuracy_score(labels_test, predictions_class_test)
print('accuracy', accuracy)

#compute precision score
precision = metrics.precision_score(labels_test, predictions_class_test)
print('precision', precision)

#compute recall score
recall = metrics.recall_score(labels_test, predictions_class_test)
print('recall', recall)

#compute f1 score
f1 = metrics.f1_score(labels_test, predictions_class_test)
print('f1', f1)


#compute false positive, true positive, auc
fpr, tpr, thresholds = metrics.roc_curve(labels_test, predictions_class_test)
auc = metrics.auc(fpr, tpr)
print('auc', auc)

#compute roc_auc_score
roc_auc_score = metrics.roc_auc_score(labels_test, predictions_class_test)
print('roc_auc_score', roc_auc_score)
