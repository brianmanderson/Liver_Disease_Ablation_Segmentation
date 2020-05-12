__author__ = 'Brian M Anderson'
# Created on 1/30/2020
from plotnine import *
import os
import numpy as np
import pandas as pd


def main():
    out_path = '.'
    excel_out_path = os.path.join(out_path, 'hparams_table.csv')
    data = pd.read_csv(excel_out_path)
    data = data.dropna()
    xxx = 1
    (ggplot(data) + aes(x='layers', y='log_epoch_loss') + facet_wrap('num_conv_blocks',
                                                                     labeller='label_both') + geom_point(
        mapping=aes(color='conv_lambda')) + xlab('layers') + ylab('Validation Loss') +
     ggtitle('Validation Loss vs Number of Layers, Number of Conv Blocks, and Conv Lambda') + scale_colour_gradient(low='blue',
                                                                                                      high='red'))
    (ggplot(data) + aes(x='layers', y='log_epoch_loss') + facet_wrap('conv_lambda', labeller='label_both') + geom_point(
        mapping=aes(color='num_conv_blocks')) + xlab('Layers') + ylab('Validation Loss') +
     ggtitle('Validation Loss vs Number of Layers, Filters, and Max Filters') + scale_colour_gradient(low='blue',
                                                                                                      high='red'))


if __name__ == '__main__':
    pass
