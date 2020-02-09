__author__ = 'Brian M Anderson'
# Created on 2/8/2020

from Base_Deeplearning_Code.Cyclical_Learning_Rate.clr_callback import Half_Drop
from Base_Deeplearning_Code.Plot_And_Scroll_Images.Plot_Scroll_Images import plot_scroll_Image, plt

lrate = Half_Drop(base_lr=1e-6, step_size=10)
iterations = []
lr = []
for i in range(200):
    iterations.append(i)
    lr.append(lrate.clr())
    lrate.trn_iterations += 1
    lrate.clr_iterations += 1
xxx = 1