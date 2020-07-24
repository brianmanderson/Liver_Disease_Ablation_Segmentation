__author__ = 'Brian M Anderson'
# Created on 7/24/2020

import pandas as pd
import os

excel_path = r'K:\Morfeus\BMAnderson\Modular_Projects\Liver_Local_Recurrence_Work\Dice_Values_and_Qualitative_Comparison_GTV_Ablation.xlsx'
excel_out_path = r'K:\Morfeus\BMAnderson\Modular_Projects\Liver_Local_Recurrence_Work\Combined_Qualitative_Results.xlsx'
odisio = pd.read_excel(excel_path, sheet_name='Odisio')
wanted_columns = ['MRN'] + [i for i in odisio.columns if i.find('Score') != -1 or i.find('Comm') != -1]
odisio = odisio[wanted_columns]

kang = pd.read_excel(excel_path, sheet_name='Kang')
wanted_columns = ['MRN'] + [i for i in kang.columns if i.find('Score') != -1 or i.find('Comm') != -1]
kang = kang[wanted_columns]
combined = pd.merge(odisio, kang)
columns = ['MRN', 'Comments GTV_Odisio', 'Comments GTV_Kang', 'Score_GTV_Odisio', 'Score_GTV_Kang',
           'Comments Ablation_Odisio', 'Comments Ablation_Kang', 'Score_Ablation_Odisio', 'Score_Ablation_Kang']
combined = combined[columns]
combined.to_excel(excel_out_path, index=0)