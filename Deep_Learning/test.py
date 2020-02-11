__author__ = 'Brian M Anderson'
# Created on 2/8/2020

from Base_Deeplearning_Code.Cyclical_Learning_Rate.clr_callback import CyclicLR, Half_Drop
from Base_Deeplearning_Code.Plot_And_Scroll_Images.Plot_Scroll_Images import plot_scroll_Image, plt

# lrate = CyclicLR(base_lr=1e-5,max_lr=1e-3,step_size=20,base_reduce_factor=2.0, mode='triangular2')
step_size = 65
step_size_factor = 50
lrate = Half_Drop(base_lr=1e-6, step_epoch=50)
iterations = []
lr = []
for i in range(200+1):
    iterations.append(i)
    lr.append(lrate.clr())
    lrate.trn_iterations += 1
    lrate.clr_iterations += 1
xxx = 1