import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import LabelEncoder
import pdb
# pdb.set_trace()

datasets = pd.read_csv("./processed_data/datasets.csv")

#checking the target classes ratio
count_classes = pd.value_counts(datasets['Class'], sort = True).sort_index()
count_classes.plot(kind = 'bar')
plt.title("Malware analysis output histogram")
plt.xlabel("Class")
plt.ylabel("Frequency")

# feature correlation matrix, 1.0 means it's highly correlated and -1.0 means it's not
fig, ax = plt.subplots(figsize=(8,8))
sns.heatmap(datasets.iloc[:,2:10].corr())

#plot a histogram for the anonymized_features
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
plt.show()
