__author__ = 'Brian M Anderson'
# Created on 1/17/2020

import os
import numpy as np
import pandas as pd


def separate_files(path):
    files = []
    for _, _, files in os.walk(path):
        break
    files = [i for i in files if i.find('Overall_mask') == 0]
    files = np.asarray(files)
    perm = np.arange(len(files))
    np.random.shuffle(perm)
    files = files[perm]
    total_len = len(files)
    train = total_len//10*6
    validation = total_len//20*3
    folders = ['Train','Validation','Test']
    out_dict = {'Image_Index':[],'Folder':[]}
    for i in folders:
        if not os.path.exists(os.path.join(path,i)):
            os.makedirs(os.path.join(path,i))
    for i in np.arange(train):
        file_name = files[i]
        image_name = file_name.replace('Overall_mask','Overall_Data')
        image_name = image_name.replace('_y','_')
        index = file_name.split('_y')[-1]
        index = index.split('.')[0]
        out_dict['Image_Index'].append(index)
        out_dict['Folder'].append('Train')
        os.rename(os.path.join(path, file_name), os.path.join(path, 'Train', file_name))
        os.rename(os.path.join(path, image_name), os.path.join(path, 'Train', image_name))
    for i in np.arange(train,train+validation):
        file_name = files[i]
        image_name = file_name.replace('Overall_mask','Overall_Data')
        image_name = image_name.replace('_y','_')
        index = file_name.split('_y')[-1]
        index = index.split('.')[0]
        out_dict['Image_Index'].append(index)
        out_dict['Folder'].append('Validation')
        os.rename(os.path.join(path, file_name), os.path.join(path, 'Validation', file_name))
        os.rename(os.path.join(path, image_name), os.path.join(path, 'Validation', image_name))
    for i in np.arange(train+validation,len(files)):
        file_name = files[i]
        image_name = file_name.replace('Overall_mask','Overall_Data')
        image_name = image_name.replace('_y','_')
        index = file_name.split('_y')[-1]
        index = index.split('.')[0]
        out_dict['Image_Index'].append(index)
        out_dict['Folder'].append('Test')
        os.rename(os.path.join(path, file_name), os.path.join(path, 'Test', file_name))
        os.rename(os.path.join(path, image_name), os.path.join(path, 'Test', image_name))
    df = pd.DataFrame()
    df = df.from_dict(out_dict)
    df.to_excel(os.path.join('..','Patient_Distribution.xlsx'))


if __name__ == '__main__':
    pass
