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
from Segmentation_Evaluation_Tools.src.SegmentationEvaluationTools import determine_sensitivity,\
    determine_false_positive_rate_and_false_volume, determine_overlap_measures


def calc_dice_single_patient(export_path, patient_dictionaries, MRN):
    if export_path.find('GTV') != -1:
        truth_handle_base = sitk.ReadImage(os.path.join(export_path, 'GTV.mhd'))
    else:
        truth_handle_base = sitk.ReadImage(os.path.join(export_path, 'Ablation.mhd'))
    prediction_handle_base = sitk.ReadImage(os.path.join(export_path, 'Liver_Disease_Ablation_BMA_Program_0.mhd'))
    files = [i for i in os.listdir(export_path) if i.endswith('.mhd')]
    liver_handle = None
    if 'Liver.mhd' in files:
        liver_handle = sitk.ReadImage(os.path.join(export_path, 'Liver.mhd'))
    elif 'Liver_BMA_Program_4.mhd' in files:
        liver_handle = sitk.ReadImage(os.path.join(export_path, 'Liver_BMA_Program_4.mhd'))
    prediction_handle_base = prediction_handle_base > 255/2
    truth_handle_base = truth_handle_base > 255/2
    if liver_handle is not None:
        liver_handle = liver_handle > 255 / 2
        prediction_handle_base *= liver_handle
        truth_handle_base *= liver_handle
    sensitivity_metrics = determine_sensitivity(prediction_handle=prediction_handle_base,
                                                truth_handle=truth_handle_base)
    fp_metrics = determine_false_positive_rate_and_false_volume(prediction_handle=prediction_handle_base,
                                                                truth_handle=truth_handle_base)
    overlap_metrics = determine_overlap_measures(prediction_handle_base=prediction_handle_base,
                                                 truth_handle_base=truth_handle_base, perform_distance_measures=True,
                                                 measure_as_multiple_sites=True)
    patient_dict = {'Sensitivity': sensitivity_metrics, 'FP': fp_metrics,
                    'Overlap': overlap_metrics}
    print(patient_dict)
    patient_dictionaries[MRN] = patient_dict


def worker_def(q):
    while True:
        item = q.get()
        if item is None:
            break
        else:
            calc_dice_single_patient(**item)
            q.task_done()


def calculate_dsc(base_export_path, excel_path='.', thread_count=int(cpu_count()*.75)):
    patient_dictionaries = {'GTV': {}, 'Ablation': {}}
    q = Queue(maxsize=thread_count)
    threads = []
    for worker in range(thread_count):
        t = Thread(target=worker_def, args=(q,))
        t.start()
        threads.append(t)
    for ext in ['GTV', 'Ablation']:
        export_path = os.path.join(base_export_path, ext)
        MRNs = os.listdir(export_path)
        for MRN in MRNs:
            print(MRN)
            path = os.path.join(export_path, MRN)
            for root, dirs, files in os.walk(path):
                if files:
                    item = {'export_path': root, 'patient_dictionaries': patient_dictionaries[ext], 'MRN': MRN}
                    # calc_dice_single_patient(**item)
                    q.put(item)
            # calc_dice_single_patient(**item)
    for i in range(thread_count):
        q.put(None)
    for t in threads:
        t.join()
    combined_dict = {}
    for ext in ['GTV', 'Ablation']:
        small_dict = patient_dictionaries[ext]
        reordered = {}
        base_key = 'Sensitivity'
        reordered[base_key] = {'Patient_ID': []}
        for MRN in small_dict.keys():
            patient_data = small_dict[MRN]
            for key in patient_data[base_key].keys():
                if key not in reordered[base_key]:
                    reordered[base_key][key] = []
                reordered[base_key][key] += patient_data[base_key][key]
            reordered[base_key]['Patient_ID'] += [MRN for _ in range(len(patient_data[base_key][key]))]
        base_key = 'FP'
        reordered[base_key] = {'Patient_ID': []}
        for MRN in small_dict.keys():
            patient_data = small_dict[MRN]
            for key in patient_data[base_key].keys():
                if key not in reordered[base_key]:
                    reordered[base_key][key] = []
                reordered[base_key][key].append(patient_data[base_key][key])
            reordered[base_key]['Patient_ID'].append(MRN)
        base_key = 'Overlap'
        reordered[base_key] = {'Patient_ID': [], 'Site': []}
        for MRN in small_dict.keys():
            patient_data = small_dict[MRN]
            for label in patient_data[base_key].keys():
                for key in patient_data[base_key][label].keys():
                    if key not in reordered[base_key]:
                        reordered[base_key][key] = []
                    reordered[base_key][key].append(patient_data[base_key][label][key])
                reordered[base_key]['Patient_ID'].append(MRN)
                reordered[base_key]['Site'].append(label)
        combined_dict[ext] = reordered
    for ext in ['GTV', 'Ablation']:
        with pd.ExcelWriter(os.path.join('.',
                                         'Raystation_Clincal_Quantitative_Results_{}.xlsx'.format(ext))) as writer:
            for key in ['Sensitivity', 'FP', 'Overlap']:
                df = pd.DataFrame(combined_dict[ext][key])
                df.to_excel(writer, sheet_name=key, index=0)


if __name__ == '__main__':
    pass
