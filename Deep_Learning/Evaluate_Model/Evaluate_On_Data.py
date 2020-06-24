__author__ = 'Brian M Anderson'
# Created on 3/4/2020

import os
import SimpleITK as sitk
import numpy as np
from enum import Enum
from Deep_Learning.Base_Deeplearning_Code.Plot_And_Scroll_Images.Plot_Scroll_Images import plot_scroll_Image
import pandas as pd
from threading import Thread
from multiprocessing import cpu_count
from queue import *


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
        self.threshold_value = seed_threshold_value
        self.Connected_Component_Filter = sitk.ConnectedComponentImageFilter()
        self.RelabelComponent = sitk.RelabelComponentImageFilter()
        self.Connected_Threshold = sitk.ConnectedThresholdImageFilter()
        self.Connected_Threshold.SetLower(lower_threshold_value)
        self.Connected_Threshold.SetUpper(2)
        self.stats = sitk.LabelShapeStatisticsImageFilter()

    def process(self, prediction):
        thresholded_image = sitk.BinaryThreshold(prediction, lowerThreshold=self.threshold_value)
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

    def process(self, A):
        out_dict_base, metric, metric_range, file = A
        pat_name = file.split('\\')[-1].split('_')[0]
        out_dict = {}
        for name, _ in OverlapMeasures.__members__.items():
            out_dict[name] = {}
        for name, _ in SurfaceDistanceMeasures.__members__.items():
            out_dict[name] = {}
        for name in out_dict.keys():
            for i in metric_range:
                out_dict[name]['{}_{}'.format(metric, i)] = []
        print(pat_name)
        truth = sitk.ReadImage(file.replace('_Image','_Truth'), sitk.sitkUInt8)
        truth_array = sitk.GetArrayFromImage(truth)
        volume = truth_array[truth_array == 1].shape[0] * np.prod(truth.GetSpacing()) / 1000
        out_dict['Volume'] = volume
        prediction = sitk.ReadImage(file.replace('_Image','_Prediction'))
        overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()

        hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()

        statistics_image_filter = sitk.StatisticsImageFilter()

        overlap_results = np.zeros((len(metric_range), len(OverlapMeasures.__members__.items())))
        surface_distance_results = np.zeros((len(metric_range), len(SurfaceDistanceMeasures.__members__.items())))

        reference_distance_map = sitk.Abs(sitk.SignedMaurerDistanceMap(truth, squaredDistance=False,
                                                                       useImageSpacing=True))
        reference_surface = sitk.LabelContour(truth)

        statistics_image_filter.Execute(reference_surface)
        num_reference_surface_pixels = int(statistics_image_filter.GetSum())
        Binary_Threshold = sitk.BinaryThresholdImageFilter()
        fill_binary = Fill_Binary_Holes()
        print(metric)
        for i, metric_value in enumerate(metric_range):
            print(metric_value)
            threshold_and_expand = Threshold_and_Expand(seed_threshold_value=.95, lower_threshold_value=metric_value)
            # Binary_Threshold.SetLowerThreshold(metric_value)
            threshold_pred = threshold_and_expand.process(prediction)
            threshold_pred = fill_binary.process(threshold_pred)
            # sitk.WriteImage(threshold_pred, file.replace('_Image', '_PredictionOutput'))
            # threshold_pred = Binary_Threshold.Execute(prediction)
            overlap_measures_filter.Execute(truth, threshold_pred)
            overlap_results[i, OverlapMeasures.jaccard.value] = overlap_measures_filter.GetJaccardCoefficient()
            overlap_results[i, OverlapMeasures.dice.value] = overlap_measures_filter.GetDiceCoefficient()
            overlap_results[i, OverlapMeasures.volume_similarity.value] = overlap_measures_filter.GetVolumeSimilarity()
            overlap_results[i, OverlapMeasures.false_negative.value] = overlap_measures_filter.GetFalseNegativeError()
            overlap_results[i, OverlapMeasures.false_positive.value] = overlap_measures_filter.GetFalsePositiveError()
            # Hausdorff distance
            hausdorff_distance_filter.Execute(truth, threshold_pred)

            surface_distance_results[
                i, SurfaceDistanceMeasures.hausdorff_distance.value] = hausdorff_distance_filter.GetHausdorffDistance()
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
            surface_distance_results[i, SurfaceDistanceMeasures.mean_surface_distance.value] = np.mean(
                all_surface_distances)
            surface_distance_results[i, SurfaceDistanceMeasures.median_surface_distance.value] = np.median(
                all_surface_distances)
            surface_distance_results[i, SurfaceDistanceMeasures.std_surface_distance.value] = np.std(
                all_surface_distances)
            surface_distance_results[i, SurfaceDistanceMeasures.max_surface_distance.value] = np.max(
                all_surface_distances)


            for _, measured_name in enumerate(OverlapMeasures):
                out_dict[measured_name.name]['{}_{}'.format(metric, metric_value)].append(
                    overlap_results[i, measured_name.value])
            for _, measured_name in enumerate(SurfaceDistanceMeasures):
                out_dict[measured_name.name]['{}_{}'.format(metric, metric_value)].append(
                    surface_distance_results[i, measured_name.value])
        out_dict_base[pat_name] = out_dict
        return out_dict_base


def worker_def(A):
    q = A[0]
    base_class = run_metrics()
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


def create_metric_chart(path = r'H:\Liver_Disease_Ablation\Predictions\Validation', out_path=os.path.join('.','Threshold'),
                        threshold_range=[0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95],
                        expand_range=[0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95],
                        desc='', thread_count=int(cpu_count()*.9-1)):
    image_list = [os.path.join(path,i) for i in os.listdir(path) if i.find('_Image') != -1]
    out_dict = {}
    for name, _ in OverlapMeasures.__members__.items():
        out_dict[name] = {'Patient_ID':[''], 'Volume': ['']}
    for name, _ in SurfaceDistanceMeasures.__members__.items():
        out_dict[name] = {'Patient_ID':[''], 'Volume': ['']}
    metric = 'Threshold'
    metric_range = threshold_range
    for name in out_dict.keys():
        for i in metric_range:
            out_dict[name]['{}_{}'.format(metric,i)] = [i]
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    new_out_dict = {}
    stuff = [new_out_dict, metric, metric_range]
    q = Queue(maxsize=thread_count)
    A = [q,]
    threads = []
    for worker in range(thread_count):
        t = Thread(target=worker_def, args=(A,))
        t.start()
        threads.append(t)
    image_list = image_list
    for file in image_list:
        item = stuff + [file]
        q.put(item)
    for i in range(thread_count):
        q.put(None)
    for t in threads:
        t.join()

    for pat_id in new_out_dict.keys():
        print(pat_id)
        for metric_name, _ in OverlapMeasures.__members__.items():
            out_dict[metric_name]['Patient_ID'].append(pat_id)
            out_dict[metric_name]['Volume'].append(new_out_dict[pat_id]['Volume'])
            for measured_name in new_out_dict[pat_id][metric_name].keys():
                out_dict[metric_name][measured_name] += new_out_dict[pat_id][metric_name][measured_name]
        for metric_name, _ in SurfaceDistanceMeasures.__members__.items():
            out_dict[metric_name]['Patient_ID'].append(pat_id)
            out_dict[metric_name]['Volume'].append(new_out_dict[pat_id]['Volume'])
            for measured_name in new_out_dict[pat_id][metric_name].keys():
                out_dict[metric_name][measured_name] += new_out_dict[pat_id][metric_name][measured_name]

    for measured_name in out_dict.keys():
        df = pd.DataFrame(out_dict[measured_name])
        df.to_excel(os.path.join(out_path,'Out_Data_{}_{}.xlsx'.format(desc,measured_name)),index=0)
    return None


if __name__ == '__main__':
    pass

