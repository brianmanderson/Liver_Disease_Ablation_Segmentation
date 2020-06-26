__author__ = 'Brian M Anderson'
# Created on 6/24/2020

import os
import SimpleITK as sitk
import numpy as np
import pandas as pd
from Deep_Learning.Base_Deeplearning_Code.Plot_And_Scroll_Images.Plot_Scroll_Images import plot_scroll_Image, plt
from threading import Thread
from multiprocessing import cpu_count
from queue import *


def calc_dice_single_patient(export_path, patient_dictionaries, MRN):
    pred_roi_base = 'Liver_Disease_Ablation_BMA_Program_0'
    overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()
    stats = sitk.LabelShapeStatisticsImageFilter()
    Connected_Component_Filter = sitk.ConnectedComponentImageFilter()
    Connected_Threshold = sitk.ConnectedThresholdImageFilter()
    Connected_Threshold.SetUpper(2)
    patient_dict = {'Primary': [], 'Primary_Volume': [], 'Secondary': [], 'Secondary_Volume': []}
    print(MRN)
    patient_path = os.path.join(export_path, MRN)
    for gt_name, exam_name in zip(['GTV_Primary.mhd', 'Ablation_Secondary.mhd'], ['Primary', 'Secondary']):
        pred_roi = pred_roi_base + '_' + exam_name + '.mhd'
        pred_primary = sitk.ReadImage(os.path.join(patient_path, pred_roi)) > 255 / 2
        gt = sitk.ReadImage(os.path.join(patient_path, gt_name)) > 255 / 2
        if os.path.exists(os.path.join(patient_path, 'Liver_Primary.mhd')):
            liver = sitk.ReadImage(os.path.join(patient_path, 'Liver_{}.mhd'.format(exam_name))) > 255 / 2
        elif os.path.exists(os.path.join(patient_path, 'Liver_BMA_Program_4_Primary.mhd')):
            liver = sitk.ReadImage(os.path.join(patient_path, 'Liver_BMA_Program_4_{}.mhd'.format(exam_name))) > 255 / 2
        pred_primary = pred_primary * liver
        gt = gt * liver
        truth_array = sitk.GetArrayFromImage(gt)
        volume = truth_array[truth_array == 1].shape[0] * np.prod(gt.GetSpacing()) / 1000
        patient_dict['{}_Volume'.format(exam_name)].append(volume)
        connected_image = Connected_Component_Filter.Execute(gt * pred_primary)
        stats.Execute(connected_image)
        seeds_gt = [stats.GetCentroid(l) for l in stats.GetLabels()]
        seeds_gt = [pred_primary.TransformPhysicalPointToIndex(i) for i in seeds_gt]

        Connected_Threshold.SetSeedList(seeds_gt)
        Connected_Threshold.SetLower(1)
        pred_grown = Connected_Threshold.Execute(pred_primary)
        overlap_measures_filter.Execute(pred_grown, gt)
        patient_dict[exam_name].append(overlap_measures_filter.GetDiceCoefficient())
    patient_dictionaries[MRN] = patient_dict


def worker_def(q):
    while True:
        item = q.get()
        if item is None:
            break
        else:
            calc_dice_single_patient(**item)
            q.task_done()


def calculate_dsc(export_path, excel_path='.', thread_count=int(cpu_count()*.75)):
    MRNs = os.listdir(export_path)
    patient_dictionaries = {}
    q = Queue(maxsize=thread_count)
    threads = []
    for worker in range(thread_count):
        t = Thread(target=worker_def, args=(q,))
        t.start()
        threads.append(t)
    for MRN in MRNs:
        item = {'export_path':export_path, 'patient_dictionaries':patient_dictionaries, 'MRN':MRN}
        q.put(item)
        # calc_dice_single_patient(**item)
    for i in range(thread_count):
        q.put(None)
    for t in threads:
        t.join()

    out_dict = {'MRN':[]}
    for MRN in patient_dictionaries:
        out_dict['MRN'].append(MRN)
        pat_dict = patient_dictionaries[MRN]
        for key in pat_dict:
            if key not in out_dict:
                out_dict[key] = []
            out_dict[key] += pat_dict[key]
    xxx = 1
    df = pd.DataFrame(out_dict)
    df.to_excel(os.path.join(excel_path,'MRN_Dice_Values_New.xlsx'),index=0)


if __name__ == '__main__':
    pass
