__author__ = 'Brian M Anderson'
# Created on 6/9/2020

import pandas as pd
import os


def fix_excel():
    excel_sheet = os.path.join('.','MRNs_All_Primary_Secondary_exam.xlsx')
    df = pd.read_excel(excel_sheet)
    df = df.to_dict()
    out_dict = {'MRN':[],'Pre':[],'Post':[]}
    for key in ['MRN']:
        for index in df[key]:
            info = str(df[key][index])
            if info.startswith('CT '):
                info = info.split(' ')[1]
            out_dict[key].append(int(info))
    for key in ['Pre','Post']:
        for index in df[key]:
            info = str(df[key][index])
            print(info)
            if info.startswith('CT '):
                info = info.split(' ')[1]
            out_dict[key].append('CT {}'.format(info))
    df = pd.DataFrame(out_dict)
    df.to_excel(excel_sheet.replace('.xlsx','_new.xlsx'),index=0)
    return None


if __name__ == '__main__':
    pass
