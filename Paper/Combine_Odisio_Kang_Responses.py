__author__ = 'Brian M Anderson'
# Created on 7/24/2020

import pandas as pd


def combine_sheets():
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
    GTV_columns = ['MRN'] + [i for i in columns if i.find('GTV') != -1]
    Ablation_columns = ['MRN'] + [i for i in columns if i.find('Ablation') != -1]
    with pd.ExcelWriter(excel_out_path) as writer:
        combined[GTV_columns].to_excel(writer, index=0, sheet_name='GTV')
        combined[Ablation_columns].to_excel(writer, index=0, sheet_name='Ablation')
    return None


if __name__ == '__main__':
    pass
