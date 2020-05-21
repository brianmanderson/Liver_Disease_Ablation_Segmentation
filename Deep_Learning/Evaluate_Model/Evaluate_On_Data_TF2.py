__author__ = 'Brian M Anderson'
# Created on 3/4/2020

import os
import SimpleITK as sitk
import numpy as np
from enum import Enum
from Deep_Learning.Base_Deeplearning_Code.Plot_And_Scroll_Images.Plot_Scroll_Images import plot_scroll_Image, plt
import pandas as pd
import pickle
from threading import Thread
from multiprocessing import cpu_count
from queue import *


def load_obj(path):
    if path.find('.pkl') == -1:
        path += '.pkl'
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    else:
        out = {}
        return out


def save_obj(path, obj): # Save almost anything.. dictionary, list, etc.
    if path.find('.pkl') == -1:
        path += '.pkl'
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.DEFAULT_PROTOCOL)
    return None


class Fill_Binary_Holes(object):
    def __init__(self, kernel_radius=(5,5,1)):
        self.BinaryfillFilter = sitk.BinaryFillholeImageFilter()
        self.BinaryfillFilter.SetFullyConnected(True)
        self.BinaryfillFilter = sitk.BinaryMorphologicalClosingImageFilter()
        self.BinaryfillFilter.SetKernelRadius(kernel_radius)
        self.BinaryfillFilter.SetKernelType(sitk.sitkBall)

    def process(self, pred_handle):
        output = self.BinaryfillFilter.Execute(pred_handle)
        return output


class Threshold_and_Expand(object):
    def __init__(self, seed_threshold_value=0.8, lower_threshold_value=0.2):
        self.seed_threshold = seed_threshold_value
        self.Connected_Component_Filter = sitk.ConnectedComponentImageFilter()
        self.Connected_Threshold = sitk.ConnectedThresholdImageFilter()
        self.Connected_Threshold.SetLower(lower_threshold_value)
        self.Connected_Threshold.SetUpper(2)
        self.stats = sitk.LabelShapeStatisticsImageFilter()

    def process(self, prediction):
        thresholded_image = sitk.BinaryThreshold(prediction, lowerThreshold=self.seed_threshold)
        connected_image = self.Connected_Component_Filter.Execute(thresholded_image)
        self.stats.Execute(connected_image)
        seeds = [self.stats.GetCentroid(l) for l in self.stats.GetLabels()]
        seeds = [thresholded_image.TransformPhysicalPointToIndex(i) for i in seeds]
        self.Connected_Threshold.SetSeedList(seeds)
        output = self.Connected_Threshold.Execute(prediction)
        return output


class OverlapMeasures(Enum):
    jaccard, dice, volume_similarity, false_negative, false_positive = range(5)


class SurfaceDistanceMeasures(Enum):
    hausdorff_distance, mean_surface_distance, median_surface_distance, std_surface_distance, max_surface_distance = range(5)


class run_metrics(object):
    def __init__(self, save_path):
        self.save_path = save_path

    def process(self, A):
        out_dict_base, threshold_range, seed_range, file = A
        pat_name = os.path.split(file)[-1]
        print(pat_name)
        truth = sitk.ReadImage(file.replace('_Image','_Truth'), sitk.sitkUInt8)
        truth_array = sitk.GetArrayFromImage(truth)
        volume = truth_array[truth_array == 1].shape[0] * np.prod(truth.GetSpacing()) / 1000
        prediction = sitk.ReadImage(file.replace('_Image','_Prediction'))
        overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()

        statistics_image_filter = sitk.StatisticsImageFilter()

        out_dict = {'volume':volume}
        for _, measured_name in enumerate(OverlapMeasures):
            out_dict[measured_name.name] = np.zeros((len(threshold_range), len(seed_range)))
        reference_surface = sitk.LabelContour(truth)
        statistics_image_filter.Execute(reference_surface)
        fill_binary = Fill_Binary_Holes()
        Connected_Component_Filter = sitk.ConnectedComponentImageFilter()
        Connected_Threshold = sitk.ConnectedThresholdImageFilter()
        Connected_Threshold.SetUpper(2)
        stats = sitk.LabelShapeStatisticsImageFilter()
        for j, seed_value in enumerate(seed_range):
            thresholded_image = sitk.BinaryThreshold(prediction, lowerThreshold=seed_value)
            connected_image = Connected_Component_Filter.Execute(thresholded_image)
            stats.Execute(connected_image)
            seeds = [stats.GetCentroid(l) for l in stats.GetLabels()]
            seeds = [thresholded_image.TransformPhysicalPointToIndex(i) for i in seeds]
            Connected_Threshold.SetSeedList(seeds)
            print('Seed value {}'.format(seed_value))
            for i, threshold_value in enumerate(threshold_range):
                print('Threshold value {}'.format(threshold_value))
                Connected_Threshold.SetLower(threshold_value)
                threshold_pred = Connected_Threshold.Execute(prediction)
                # threshold_pred = fill_binary.process(threshold_pred)
                overlap_measures_filter.Execute(truth, threshold_pred)
                out_dict[OverlapMeasures.jaccard.name][i, j] = overlap_measures_filter.GetJaccardCoefficient()
                out_dict[OverlapMeasures.dice.name][i, j] = overlap_measures_filter.GetDiceCoefficient()
                out_dict[OverlapMeasures.volume_similarity.name][i, j] = overlap_measures_filter.GetVolumeSimilarity()
                out_dict[OverlapMeasures.false_negative.name][i, j] = overlap_measures_filter.GetFalseNegativeError()
                out_dict[OverlapMeasures.false_positive.name][i, j] = overlap_measures_filter.GetFalsePositiveError()
        save_obj(os.path.join(self.save_path,'{}_out_dict.pkl'.format(pat_name)),out_dict)
        return out_dict_base


def worker_def(A):
    q, save_path = A
    base_class = run_metrics(save_path)
    while True:
        item = q.get()
        if item is None:
            break
        else:
            try:
                base_class.process(item)
            except:
                print('failed?')
            q.task_done()


def create_metric_chart(path = r'D:\Liver_Disease_Ablation\Predictions\Validation', out_path=os.path.join('.','Threshold'),
                        threshold_range=[0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95],
                        seed_range=[0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95],
                        desc='', thread_count=int(cpu_count()*.95-1), re_write=False):
    image_list = [os.path.join(path,i) for i in os.listdir(path) if i.find('_Image') != -1]
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    new_out_dict = load_obj(os.path.join('.','out_dict_{}.pkl'.format(desc)))
    stuff = [new_out_dict, threshold_range, seed_range]
    q = Queue(maxsize=thread_count)
    A = [q,out_path]
    threads = []
    for worker in range(thread_count):
        t = Thread(target=worker_def, args=(A,))
        t.start()
        threads.append(t)
    image_list = image_list
    for file in image_list:
        pat_name = os.path.split(file)[-1]
        if os.path.exists(os.path.join(out_path, '{}_out_dict.pkl'.format(pat_name))) and not re_write:
            continue
        item = stuff + [file]
        q.put(item)
    for i in range(thread_count):
        q.put(None)
    for t in threads:
        t.join()
    out_dict = {}
    for name, _ in OverlapMeasures.__members__.items():
        out_dict[name] = np.zeros((len(image_list), len(threshold_range), len(seed_range)))
    out_dict['volume'] = np.zeros(len(image_list))
    for i, file in enumerate(image_list):
        pat_name = os.path.split(file)[-1]
        if os.path.exists(os.path.join(out_path, '{}_out_dict.pkl'.format(pat_name))):
            patient_dict = load_obj(os.path.join(out_path, '{}_out_dict.pkl'.format(pat_name)))
            for key in patient_dict:
                out_dict[key][i] = patient_dict[key]
    dice = out_dict['dice']
    volume = out_dict['volume']
    average_dice = np.mean(dice[volume > 10], axis=0) # take the average across all patients
    xxx = 1
    return None


if __name__ == '__main__':
    pass

