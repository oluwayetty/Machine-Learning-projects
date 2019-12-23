import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import LabelEncoder
import numpy as np
from lg import datasets, clf, features_train, y_resampled
import pdb
# pdb.set_trace()

#checking the target classes ratio for original datasets
count_classes = pd.value_counts(datasets['Class'], sort = True).sort_index()
count_classes.plot(kind = 'bar')
plt.title("Malware analysis output histogram")
plt.xlabel("Class")
plt.ylabel("Frequency")

#checking the target classes ratio for resampled data
classes = pd.value_counts(y_resampled, sort = True).sort_index()
classes.plot(kind = 'bar')
plt.title("Malware analysis output histogram for resampled data")
plt.xlabel("Output")
plt.ylabel("Fre")

# feature correlation matrix, 1.0 means it's highly correlated and -1.0 means it's not
fig, ax = plt.subplots(figsize=(8,8))
sns.heatmap(datasets.iloc[:,2:10].corr())

#plot a histogram for the features
features = datasets.iloc[:,2:10].columns

plt.figure(figsize=(15,30))
gs = gridspec.GridSpec(15, 4)
for i, cn in enumerate(datasets[features]):
    ax = plt.subplot(gs[i])
    sns.distplot(datasets[cn][datasets.Class == True], bins=30)
    sns.distplot(datasets[cn][datasets.Class == False], bins=30)
    ax.set_xlabel('')
    ax.set_title('histogram of feature: ' + str(cn))
plt.tight_layout()

# feature importance
importances = clf.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Plot the feature importances of the forest
print("Feature ranking:")
plt.figure(figsize=(14, 6))
plt.title("Feature importances")
plt.bar(range(features_train.shape[1]), importances[indices],
       color="b", yerr=std[indices], align="center")
mapIndex = map(lambda x: features_train,list(indices))
plt.xticks(range(features_train.shape[1]), mapIndex)
plt.xlim([-1, features_train.shape[1]])
plt.show()
