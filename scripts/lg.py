"""
    Let's build our model :)
"""
import pandas as pd
import numpy as np
import pdb
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression as LG
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn import tree
from sklearn.naive_bayes import BernoulliNB as NB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score as acc
from sklearn.metrics import precision_score as precision
from sklearn.metrics import recall_score as recall
from sklearn.metrics import f1_score as f1

datasets = pd.read_csv("./processed_data/datasets.csv")

# encoding our class value [True =1. False = 0]
label_encoder = LabelEncoder()
class_integer_encoded  = label_encoder.fit_transform(datasets['Class'])

# splitting my data in ratio of 30:70 percent
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(datasets.iloc[:,2:10], class_integer_encoded, test_size=0.3, random_state=0)

# train our model
# tree.DecisionTreeClassifier().fit()
clf = RFC().fit(features_train, labels_train)

# use our model to predict
predictions_class_test = clf.predict(features_test)

#compute accuracy_score
accuracy = acc(labels_test, predictions_class_test)
print('accuracy', accuracy)

#compute precision score
precision = precision(labels_test, predictions_class_test)
print('precision', precision)

#compute recall score
recall = recall(labels_test, predictions_class_test)
print('recall', recall)

#compute f1 score
f1 = f1(labels_test, predictions_class_test)
print('f1', f1)
