import shutil
import os

def copy_torch(dir, new_dir):
    if not os.path.exists(new_dir):
        os.mkdir(new_dir)
    with open(os.path.join(dir, 'ground_truth.txt'), 'r') as file:
        # import pdb; pdb.set_trace()
        for line in file:
            filename, label = tuple(filter(None, line.split(';')))
            label = label.replace('\n', '')
            path = os.path.join(new_dir, label)
            if not os.path.exists(path):
                os.mkdir(path)
            shutil.copy2(os.path.join(dir, filename), os.path.join(path, filename))
    return


if __name__ == '__main__':
    copy_torch('../data/venice_testdata', '../data/classified_venice_testdata')
