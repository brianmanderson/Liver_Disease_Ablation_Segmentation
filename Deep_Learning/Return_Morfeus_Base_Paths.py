__author__ = 'Brian M Anderson'
# Created on 2/17/2020

import os
from Base_Deeplearning_Code.Data_Generators.Return_Paths import find_base_dir


def return_paths():
    try:
        base = r'\\mymdafiles\di_data1'
        base_path = r'D:\Liver_Disease_Ablation'
        os.listdir(base_path)
    except:
        base = find_base_dir()
        base_path = os.path.join(base, 'Liver_GTV_Ablation')
    morfeus_drive = os.path.abspath(os.path.join(base,'Morfeus','BMAnderson','Modular_Projects','Liver_Disease_Ablation_Segmentation_Work'))
    return base_path, morfeus_drive


if __name__ == '__main__':
    pass
