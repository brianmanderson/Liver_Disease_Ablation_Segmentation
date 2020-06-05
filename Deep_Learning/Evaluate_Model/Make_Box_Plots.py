__author__ = 'Brian M Anderson'
# Created on 3/4/2020

import matplotlib.pyplot as plt
import pickle, os


def load_obj(path):
    if path.find('.pkl') == -1:
        path += '.pkl'
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    else:
        out = {}
        return out


def create_plot(title, values, metric, out_path=None, y_ticks=None):
    x_ticks = ['']
    num_labels = [i for i in range(len(x_ticks))]
    plt.figure(0)
    plt.suptitle(title)
    ax = plt.subplot(1,1,1)
    plt.boxplot(values)
    plt.xlabel('LiTs Test Set')
    plt.ylabel(metric)
    plt.xticks(num_labels,x_ticks)
    if y_ticks is not None:
        plt.yticks(y_ticks)
    if out_path is not None:
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        plt.savefig(os.path.join(out_path, '{}_{}.jpg'.format(title, metric)), quality=95)
    else:
        plt.show()
    return None


if __name__ == '__main__':
    pass
