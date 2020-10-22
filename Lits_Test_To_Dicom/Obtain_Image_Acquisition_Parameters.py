__author__ = 'Brian M Anderson'
# Created on 10/22/2020
import os
import SimpleITK as sitk
import pandas as pd


def obtain_image_parameters(path=r'H:\Liver_Disease_Ablation\LiTs_Test\Nifti',
                            excel_path=os.path.join('.', 'Image_Parameters.xlsx')):
    files = [os.path.join(path, i) for i in os.listdir(path) if i.startswith('test-volume') and i.find('liver') == -1]

    reader = sitk.ImageFileReader()
    out_dict = {'file': [], 'x': [], 'y': [], 'thickness': []}
    for file in files:
        print(file)
        reader.SetFileName(file)
        reader.Execute()
        spacing = reader.GetSpacing()
        out_dict['x'].append(spacing[0])
        out_dict['y'].append(spacing[1])
        out_dict['thickness'].append(spacing[2])
        out_dict['file'].append(os.path.split(file)[-1])
    df = pd.DataFrame(out_dict)
    df.to_excel(excel_path, index=0)
    return None


if __name__ == '__main__':
    pass
