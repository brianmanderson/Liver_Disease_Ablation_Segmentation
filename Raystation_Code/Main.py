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

write_csv = True
if write_csv:
    df = pd.read_excel(os.path.join('.','MRNs_All_Primary_Secondary_exam.xlsx'))
    df.to_csv(os.path.join('.','MRNs_All_Primary_Secondary_exam.csv'), index=0)

'''
Now, run Predict_Disease_Ablation_From_CSV in Raystation
'''