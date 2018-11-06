"""
    Let's build our model :)
"""
import pandas as pd
import pdb

from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression as LG
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score as acc
from sklearn.metrics import precision_score as precision
from sklearn.metrics import recall_score as recall

training_data = pd.read_csv("./processed_data/feature_vectors_data.csv")

# encoding our output value [True =1. False = 0]
label_encoder = LabelEncoder()
output_integer_encoded  = label_encoder.fit_transform(training_data['output'])

# splitting my data in ratio of 30:70 percent
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(training_data.iloc[:,2:10], output_integer_encoded, test_size=0.3, random_state=0)

# train our model with LogisticRegression
clf = LG().fit(features_train, labels_train)

# use our model to predict
predictions_class_test = clf.predict(features_test)
# print(predictions_class_test)

#compute accuracy_score
accuracy = acc(labels_test, predictions_class_test)
print('accuracy', accuracy)

#compute precision score
precision = precision(labels_test, predictions_class_test)
print('precision', precision)

#compute precision score
recall = recall(labels_test, predictions_class_test)
print('recall', recall)
