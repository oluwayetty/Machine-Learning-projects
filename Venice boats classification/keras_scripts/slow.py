import os

venice_traindata_path = "../Venice-boat-classification/drive-data/venice_traindata"
venice_traindata = ([name for name in os.listdir(venice_traindata_path) if os.path.isdir(os.path.join(venice_traindata_path, name))]) # get all directories

venice_testdata_path = "../Venice-boat-classification/drive-data/classified_venice_testdata"
venice_testdata = ([name for name in os.listdir(venice_testdata_path) if os.path.isdir(os.path.join(venice_testdata_path, name))])

for folder in venice_traindata:
    contents = os.listdir(os.path.join(venice_traindata_path,folder))
    print(folder, ":", len(contents))

for folder in venice_testdata:
    contents = os.listdir(os.path.join(venice_testdata_path,folder))
    print(folder, ":", len(contents))


# from mlxtend.evaluate import confusion_matrix
# from mlxtend.plotting import plot_confusion_matrix
# import matplotlib.pyplot as plt
#
# y_target =    [1, 1, 1, 0, 0, 2, 0, 3]
# y_predicted = [1, 0, 1, 0, 0, 2, 1, 3]
#
# cm = confusion_matrix(y_target=y_target,
#                       y_predicted=y_predicted,
#                       binary=False)
#
# fig, ax = plot_confusion_matrix(conf_mat=cm)
# plt.show()
