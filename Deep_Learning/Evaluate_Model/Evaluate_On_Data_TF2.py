__author__ = 'Brian M Anderson'
# Created on 3/4/2020

import os
import SimpleITK as sitk
import numpy as np
from enum import Enum
from Deep_Learning.Base_Deeplearning_Code.Plot_And_Scroll_Images.Plot_Scroll_Images import plot_scroll_Image, plt
from Base_Deeplearning_Code.Data_Generators.Image_Processors_Module.Resample_Class.Resample_Class import Resample_Class_Object
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

    def process(self, threshold_range, seed_range, file, write_final_prediction=False):
        resampler = Resample_Class_Object()
        reader = sitk.ImageFileReader()
        reader.LoadPrivateTagsOn()
        pat_name = os.path.split(file)[-1].split('.')[0]
        print(pat_name)
        truth = sitk.ReadImage(file.replace('_Image','_Truth'), sitk.sitkUInt8)
        mask = sitk.ReadImage(file.replace('_Image','_Mask'), sitk.sitkUInt8)
        mask_filter = sitk.MaskImageFilter()
        mask_filter.SetMaskingValue(0)
        truth_array = sitk.GetArrayFromImage(truth)
        volume = truth_array[truth_array == 1].shape[0] * np.prod(truth.GetSpacing()) / 1000
        prediction = sitk.ReadImage(file.replace('_Image','_Prediction'))
        prediction = resampler.resample_image(prediction, ref_handle=truth)
        prediction = mask_filter.Execute(prediction, mask)
        overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()

        statistics_image_filter = sitk.StatisticsImageFilter()

        hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()

        out_dict = {'volume':volume}
        for _, measured_name in enumerate(OverlapMeasures):
            out_dict[measured_name.name] = np.zeros((len(threshold_range), len(seed_range)))
        for _, measured_name in enumerate(SurfaceDistanceMeasures):
            out_dict[measured_name.name] = np.ones((len(threshold_range), len(seed_range)))*999
        reference_surface = sitk.LabelContour(truth)
        reference_distance_map = sitk.Abs(sitk.SignedMaurerDistanceMap(truth, squaredDistance=False,
                                                                       useImageSpacing=True))
        statistics_image_filter.Execute(reference_surface)
        fill_binary = Fill_Binary_Holes()
        Connected_Component_Filter = sitk.ConnectedComponentImageFilter()
        Connected_Threshold = sitk.ConnectedThresholdImageFilter()
        Connected_Threshold.SetUpper(2)
        stats = sitk.LabelShapeStatisticsImageFilter()
        num_reference_surface_pixels = int(statistics_image_filter.GetSum())
        for j, seed_value in enumerate(seed_range):
            thresholded_image = sitk.BinaryThreshold(prediction, lowerThreshold=seed_value)
            connected_image = Connected_Component_Filter.Execute(thresholded_image)
            stats.Execute(connected_image)
            seeds = [stats.GetCentroid(l) for l in stats.GetLabels()]
            seeds = [thresholded_image.TransformPhysicalPointToIndex(i) for i in seeds]
            Connected_Threshold.SetSeedList(seeds)
            print('Seed value {}'.format(seed_value))
            for i, threshold_value in enumerate(threshold_range):
                # print('Threshold value {}'.format(threshold_value))
                Connected_Threshold.SetLower(threshold_value)
                threshold_pred = Connected_Threshold.Execute(prediction)
                if write_final_prediction:
                    threshold_pred = fill_binary.process(threshold_pred)
                    threshold_pred.SetOrigin(truth.GetOrigin())
                    threshold_pred.SetDirection(truth.GetDirection())
                    sitk.WriteImage(threshold_pred, file.replace('_Image','_FinalPrediction'))
                overlap_measures_filter.Execute(truth, threshold_pred)
                out_dict[OverlapMeasures.jaccard.name][i, j] = overlap_measures_filter.GetJaccardCoefficient()
                out_dict[OverlapMeasures.dice.name][i, j] = overlap_measures_filter.GetDiceCoefficient()
                out_dict[OverlapMeasures.volume_similarity.name][i, j] = overlap_measures_filter.GetVolumeSimilarity()
                out_dict[OverlapMeasures.false_negative.name][i, j] = overlap_measures_filter.GetFalseNegativeError()
                out_dict[OverlapMeasures.false_positive.name][i, j] = overlap_measures_filter.GetFalsePositiveError()

                # Symmetric surface distance measures
                segmented_distance_map = sitk.Abs(
                    sitk.SignedMaurerDistanceMap(threshold_pred, squaredDistance=False, useImageSpacing=True))
                segmented_surface = sitk.LabelContour(threshold_pred)

                # Multiply the binary surface segmentations with the distance maps. The resulting distance
                # maps contain non-zero values only on the surface (they can also contain zero on the surface)
                seg2ref_distance_map = reference_distance_map * sitk.Cast(segmented_surface, sitk.sitkFloat32)
                ref2seg_distance_map = segmented_distance_map * sitk.Cast(reference_surface, sitk.sitkFloat32)

                # Get the number of pixels in the reference surface by counting all pixels that are 1.
                statistics_image_filter.Execute(segmented_surface)
                num_segmented_surface_pixels = int(statistics_image_filter.GetSum())

                # Get all non-zero distances and then add zero distances if required.
                seg2ref_distance_map_arr = sitk.GetArrayViewFromImage(seg2ref_distance_map)
                seg2ref_distances = list(seg2ref_distance_map_arr[seg2ref_distance_map_arr != 0])
                seg2ref_distances = seg2ref_distances + \
                                    list(np.zeros(num_segmented_surface_pixels - len(seg2ref_distances)))
                ref2seg_distance_map_arr = sitk.GetArrayViewFromImage(ref2seg_distance_map)
                ref2seg_distances = list(ref2seg_distance_map_arr[ref2seg_distance_map_arr != 0])
                ref2seg_distances = ref2seg_distances + \
                                    list(np.zeros(num_reference_surface_pixels - len(ref2seg_distances)))

                all_surface_distances = seg2ref_distances + ref2seg_distances

                # The maximum of the symmetric surface distances is the Hausdorff distance between the surfaces. In
                # general, it is not equal to the Hausdorff distance between all voxel/pixel points of the two
                # segmentations, though in our case it is. More on this below.

                try:
                    hausdorff_distance_filter.Execute(truth, threshold_pred)
                    out_dict[SurfaceDistanceMeasures.hausdorff_distance.name][
                        i, j] = hausdorff_distance_filter.GetHausdorffDistance()
                except:
                    out_dict[SurfaceDistanceMeasures.hausdorff_distance.name][
                        i, j] = 9999
                out_dict[SurfaceDistanceMeasures.mean_surface_distance.name][i, j] = np.mean(
                    all_surface_distances)

                out_dict[SurfaceDistanceMeasures.median_surface_distance.name][i, j] = np.median(
                    all_surface_distances)
                out_dict[SurfaceDistanceMeasures.std_surface_distance.name][i, j] = np.std(
                    all_surface_distances)
                out_dict[SurfaceDistanceMeasures.max_surface_distance.name][i, j] = np.max(
                    all_surface_distances)
        save_obj(os.path.join(self.save_path,'{}_out_dict.pkl'.format(pat_name)),out_dict)


def worker_def(A):
    q, save_path = A
    base_class = run_metrics(save_path)
    while True:
        item = q.get()
        if item is None:
            break
        else:
            try:
                base_class.process(**item)
            except:
                print('failed?')
            q.task_done()


def combine_patient_pickles(out_path):
    image_list = [os.path.join(out_path, i) for i in os.listdir(out_path) if i.find('_out_dict.pkl') != -1]
    out_dict = {'volume':[],'patient_name':[]}
    for file in image_list:
        pat_name = os.path.split(file)[-1].split('_out_dict')[0]
        out_dict['patient_name'].append(pat_name)
        patient_dict = load_obj(file)
        for key in patient_dict:
            if key not in out_dict:
                out_dict[key] = []
            out_dict[key].append(patient_dict[key])
    for key in out_dict.keys():
        out_dict[key] = np.asarray(out_dict[key])
    return out_dict


def find_best_threshold_seed(threshold_range, seed_range, out_path):
    out_dict = combine_patient_pickles(out_path)
    threshold_names = ['Threshold_{}'.format(i) for i in threshold_range]
    seed_names = ['Seed_{}'.format(i) for i in seed_range]
    volume = out_dict['volume']
    mask = volume>20
    patient_names = out_dict['patient_name'][mask]
    for key in out_dict.keys():
        if key == 'volume' or key == 'patient_name':
            continue
        data = np.median(out_dict[key][mask], axis=0)
        if key in ['jaccard','dice', 'volume_similarity']:
            threshold, seed = np.where(np.round(data,4) == np.max(np.round(data,4)))
            title = 'Max'
        else:
            title = 'Min'
            threshold, seed = np.where(np.round(data,4) == np.min(np.round(data,4)))
        threshold, seed = np.unique(threshold), np.unique(seed)
        threshold, seed = threshold[len(threshold)//2], seed[len(seed)//2]
        print(
            '{} {} is {} at threshold of {} and seed of {}'.format(title, key, np.round(data[threshold,seed],3),
                                                                   np.round(threshold_range[threshold],2),
                                                                   np.round(seed_range[seed],2)))
        df = pd.DataFrame(data=data, index=threshold_names, columns=seed_names)
        df.to_excel(os.path.join(out_path,'{}.xlsx'.format(key)))
    return None


def create_metric_chart(path = r'D:\Liver_Disease_Ablation\Predictions\Validation', out_path=os.path.join('.','Threshold'),
                        threshold_range=[0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95],
                        seed_range=[0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95],
                        thread_count=int(cpu_count()*.95-1), re_write=False, write_final_prediction=False):
    image_list = [os.path.join(path,i) for i in os.listdir(path) if i.find('_Image') != -1]
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    q = Queue(maxsize=thread_count)
    A = [q,out_path]
    threads = []
    for worker in range(thread_count):
        t = Thread(target=worker_def, args=(A,))
        t.start()
        threads.append(t)
    for file in image_list:
        pat_name = os.path.split(file)[-1].split('.')[0]
        if os.path.exists(os.path.join(out_path, '{}_out_dict.pkl'.format(pat_name))) and not re_write:
            continue
        item = {'threshold_range': threshold_range, 'seed_range': seed_range,
                'write_final_prediction': write_final_prediction, 'file':file}
        q.put(item)
    for i in range(thread_count):
        q.put(None)
    for t in threads:
        t.join()

    if not write_final_prediction:
        find_best_threshold_seed(threshold_range=threshold_range, seed_range=seed_range, out_path=out_path)
        return None
    out_dict = combine_patient_pickles(out_path)
    for key in out_dict.keys():
        out_dict[key] = list(np.squeeze(out_dict[key]))
    df = pd.DataFrame(data=out_dict)
    df.to_excel(os.path.join(out_path, 'Final_Prediction.xlsx'), index=0)



if __name__ == '__main__':
    pass

