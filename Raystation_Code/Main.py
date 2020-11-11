__author__ = 'Brian M Anderson'
# Created on 6/9/2020

import os
import pandas as pd
'''
First, Fix_Excel to make MRNs, Pre, Post have CT #, like RS
'''
fix_excel = False
if fix_excel:
    from Raystation_Code.Fix_Excel import fix_excel
    fix_excel()

write_csv = False
if write_csv:
    df = pd.read_excel(os.path.join('.','MRNs_All_Primary_Secondary_exam.xlsx'))
    df.to_csv(os.path.join('.','MRNs_All_Primary_Secondary_exam.csv'), index=0)

'''
Now, run Predict_Disease_Ablation_From_CSV in Raystation and export to path
'''

export_path = r'H:\Single_Site'

calculate_dsc = True
if calculate_dsc:
    from Raystation_Code.Calculate_Dice import calculate_dsc
    calculate_dsc(base_export_path=export_path, excel_path='.')

make_box_plots = False
if make_box_plots:
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib, os
    import numpy as np
    df = pd.read_excel(os.path.join('.','MRN_Dice_Values.xlsx'), sheet_name='Curated')
    gtv_data = df['Primary'].values
    gtv_data = gtv_data[~np.isnan(gtv_data)]
    ablation_data = df['Secondary'].values
    ablation_data = ablation_data[~np.isnan(ablation_data)]
    matplotlib.rc('xtick', labelsize=16)
    matplotlib.rc('ytick', labelsize=16)
    title_font = {'fontname': 'Arial', 'size': '20', 'color': 'black', 'weight': 'normal',
                  'verticalalignment': 'bottom'}
    axis_font = {'fontname': 'Arial', 'size': '16', 'color': 'black', 'weight': 'normal'}
    plt.figure()
    plt.title('GTV Dice Similarity Coefficient', **title_font)
    plt.boxplot(gtv_data)
    plt.xlabel('GTV', **axis_font)
    plt.ylabel('Dice Similarity Coefficient', **axis_font)
    plt.xticks([i for i in range(1)],[''])
    plt.yticks([0.,0.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0])
    # plt.savefig(image_path, quality=95)
    plt.show()

    plt.title('Ablation Dice Similarity Coefficient', **title_font)
    plt.ylabel('Dice Similarity Coefficient', **axis_font)
    plt.yticks([0., 0.1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0])
    plt.boxplot(ablation_data)
    plt.xlabel('Ablation', **axis_font)
    plt.xticks([i for i in range(1)],[''])
    plt.show()
    xxx = 1