__author__ = 'Brian M Anderson'
# Created on 3/4/2020

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle, os
import SimpleITK as sitk


def load_obj(path):
    if path.find('.pkl') == -1:
        path += '.pkl'
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    else:
        out = {}
        return out

path = r'C:\Users\bmanderson\Desktop\Modular_Projects\Liver_Disease_Ablation_Segmentation\Deep_Learning' \
       r'\Evaluate_Model\Test_Output\Out_Data_FWHM_dice.xlsx'


data = pd.read_excel(path)
data = data.to_dict()
dice_values = np.asarray(list(data['Dice'].values()))
volume_values = np.asarray(list(data['Volume'].values()))
less_than_ten_cc_line = volume_values <= 10
greater_than_ten_cc_line = volume_values > 10

for title, values in zip(['Disease Dice > 10 cc','Disease Dice <= 10 cc'],[greater_than_ten_cc_line,less_than_ten_cc_line]):
       values = [dice_values[values]]
       x_ticks = ['']
       num_labels = [i for i in range(len(x_ticks))]
       # 0,.1,.2,.3,.4,.5,.6,.7,
       y_ticks = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
       plt.figure(0)
       plt.suptitle(title)
       ax = plt.subplot(1,1,1)
       metric = 'Overlap_Results'
       plt.boxplot(values)
       plt.xlabel('LiTs Test Set')
       plt.ylabel('Dice Simiarlity Coefficient')
       plt.xticks(num_labels,x_ticks)
       plt.yticks(y_ticks)
       plt.show()
