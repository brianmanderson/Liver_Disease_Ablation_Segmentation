__author__ = 'Brian M Anderson'
# Created on 7/28/2020

from Base_Deeplearning_Code.Cyclical_Learning_Rate.clr_callback_TF2 import CyclicLR
from Base_Deeplearning_Code.Plot_And_Scroll_Images.Plot_Scroll_Images import plot_scroll_Image
from matplotlib import pyplot as plt

min_lr = 1e-10
max_lr = 1e-9
step_size = 5
step_size_factor = 5
lrate = CyclicLR(base_lr=min_lr, max_lr=max_lr, step_size=step_size, step_size_factor=step_size_factor,
                 mode='triangular2', pre_cycle=0, base_reduce_factor=1,
                 step_size_factor_scale=lambda x: x + 0, reduction_factor=2)
indexes = []
lr = []
for i in range(200):
    lr.append(lrate.clr())
    lrate.trn_iterations += 1
    lrate.clr_iterations += 1
    indexes.append(i)
title_font = {'fontname': 'Arial', 'size': '24', 'color': 'black'}
axis_font = {'fontname': 'Arial', 'size': '20', 'color': 'black'}
plt.figure()
plt.plot(indexes, lr)
plt.yscale('log')
plt.ylabel('Log Learning Rate', **axis_font)
plt.title('Log Learning Rate vs Batch step', **title_font)
plt.xlabel('Batch Step', **axis_font)
