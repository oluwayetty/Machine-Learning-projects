import re
import string
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torchvision


# replace multiple strings
def replace(string, substitutions):
    substrings = sorted(substitutions, key=len, reverse=True)
    regex = re.compile('|'.join(map(re.escape, substrings)))
    return regex.sub(lambda match: substitutions[match.group(0)], string)


def clean():
    filenames = []
    labels = []
    with open('../data/venice_testdata/ground_truth.txt', 'r') as file:
        for line in file:
            import pdb; pdb.set_trace()

            filename, label = tuple(filter(None, line.split(';')))
            label = replace(label, {'Snapshot': '', '\n': '', ' ': ''})
            label = label.translate({ord(c): None for c in string.punctuation})
            filenames.append(filename)
            labels.append(label)

    with open('../data/venice_testdata/ground_truth.txt', 'w') as dataset:
        for file, label in zip(filenames, labels):
            dataset.write(file + ';' + label + '\n')
    return filenames, labels

clean()

if __name__ == "__main__":
    clean()
