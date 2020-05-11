__author__ = 'Brian M Anderson'
# Created on 1/30/2020
from plotnine import *
import os
import pandas as pd


def main():
    out_path = '.'
    excel_out_path = os.path.join(out_path, 'hparams_table.csv')
    data = pd.read_csv(excel_out_path)
    data = data.dropna()
    (ggplot(data) + aes(x='layers',y='epoch_loss') + facet_wrap('filters',labeller='label_both') + geom_point(mapping=aes(color='max_filters')) + xlab('Layers') + ylab('Validation Dice') +
     ggtitle('Validation Dice vs Number of Layers, Filters, and Max Filters')+scale_colour_gradient(low='blue',high='red'))
    (ggplot(data) + aes(x='layers',y='val_loss') + facet_wrap('filters',labeller='label_both') + geom_point(mapping=aes(color='max_filters')) + xlab('Layers') + ylab('Validation Dice') +
     ggtitle('Validation Loss vs Number of Layers, Filters, and Max Filters')+scale_colour_gradient(low='blue',high='red')+scale_y_log10())


if __name__ == '__main__':
    pass
