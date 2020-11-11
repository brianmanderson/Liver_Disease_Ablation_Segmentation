__author__ = 'Brian M Anderson'
# Created on 3/4/2020

import os, copy
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
    mean_surface_distance, median_surface_distance, std_surface_distance, max_surface_distance = range(4)


class run_index_single_disease(object):
    def __init__(self):
        xxx = 1

    def process(self, base_index, seed_value, threshold_value, prediction, connected_truth, out_dict, percentage,
                tumor_labels, write_final_prediction=False):
        # start = time.time()
        prediction_handle = sitk.GetImageFromArray(prediction)
        prediction_handle.SetSpacing(connected_truth.GetSpacing())
        prediction_handle.SetOrigin(connected_truth.GetOrigin())
        prediction_handle.SetDirection(connected_truth.GetDirection())
        Connected_Component_Filter = sitk.ConnectedComponentImageFilter()
        Connected_Threshold = sitk.ConnectedThresholdImageFilter()
        Connected_Threshold.SetUpper(2)

        overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()

        pred_stats = sitk.LabelShapeStatisticsImageFilter()
        thresholded_image = sitk.BinaryThreshold(prediction_handle, lowerThreshold=seed_value)
        connected_image = Connected_Component_Filter.Execute(thresholded_image)
        pred_stats.Execute(connected_image)
        pred_seeds = [pred_stats.GetCentroid(l) for l in pred_stats.GetLabels()]
        pred_seeds = [thresholded_image.TransformPhysicalPointToIndex(i) for i in pred_seeds]
        Connected_Threshold.SetSeedList(pred_seeds)
        Connected_Threshold_for_pred = sitk.ConnectedThresholdImageFilter()
        Connected_Threshold_for_pred.SetUpper(2)
        Connected_Threshold_for_pred.SetLower(1)

        Connected_Threshold.SetLower(threshold_value)
        threshold_pred_base = Connected_Threshold.Execute(prediction_handle)
        truth_stats = sitk.LabelShapeStatisticsImageFilter()

        for tumor_index, tumor_label in enumerate(tumor_labels):
            index = tuple(base_index + [tumor_index])
            truth = connected_truth == tumor_label
            truth_stats.Execute(truth)
            overlap = sitk.GetArrayFromImage(truth) * prediction
            if np.max(overlap) == 0:  # If there is no overlap, take the closest one
                seeds = [truth_stats.GetCentroid(l) for l in truth_stats.GetLabels()]
                seeds = [thresholded_image.TransformPhysicalPointToIndex(i) for i in seeds]
                seeds = [pred_seeds[np.argmin(np.sqrt(np.sum(np.subtract(pred_seeds, seeds[0])**2, axis=-1)), axis=-1)]]
            else:
                seeds = np.transpose(np.asarray(np.where(overlap > 0)))[..., ::-1]
                seeds = [[int(i) for i in j] for j in seeds]
            Connected_Threshold_for_pred.SetSeedList(seeds)
            threshold_pred = Connected_Threshold_for_pred.Execute(threshold_pred_base)
            overlap_measures_filter.Execute(truth, threshold_pred)
            out_dict[OverlapMeasures.jaccard.name][index] = overlap_measures_filter.GetJaccardCoefficient()
            out_dict[OverlapMeasures.dice.name][index] = overlap_measures_filter.GetDiceCoefficient()
            out_dict[OverlapMeasures.volume_similarity.name][index] = overlap_measures_filter.GetVolumeSimilarity()
            out_dict[OverlapMeasures.false_negative.name][index] = overlap_measures_filter.GetFalseNegativeError()
            out_dict[OverlapMeasures.false_positive.name][index] = overlap_measures_filter.GetFalsePositiveError()
            if write_final_prediction:
                statistics_image_filter = sitk.StatisticsImageFilter()
                reference_surface = sitk.LabelContour(truth)
                reference_distance_map = sitk.Abs(sitk.SignedMaurerDistanceMap(truth, squaredDistance=False,
                                                                               useImageSpacing=True))
                statistics_image_filter.Execute(reference_surface)
                num_reference_surface_pixels = int(statistics_image_filter.GetSum())
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

                out_dict[SurfaceDistanceMeasures.mean_surface_distance.name][index] = np.mean(
                    all_surface_distances)

                out_dict[SurfaceDistanceMeasures.median_surface_distance.name][index] = np.median(
                    all_surface_distances)
                out_dict[SurfaceDistanceMeasures.std_surface_distance.name][index] = np.std(
                    all_surface_distances)
                out_dict[SurfaceDistanceMeasures.max_surface_distance.name][index] = np.max(
                    all_surface_distances)
            # print(out_dict[OverlapMeasures.dice.name][index])
        print('{}% Done'.format(percentage))
        return None


class run_metrics_single_patient(object):
    def __init__(self, save_path):
        self.save_path = save_path

    def process(self, threshold_range, seed_range, file, write_final_prediction=False):
        resampler = Resample_Class_Object()
        reader = sitk.ImageFileReader()
        reader.LoadPrivateTagsOn()
        pat_name = os.path.split(file)[-1].split('.')[0]
        print(pat_name)
        truth = sitk.ReadImage(file.replace('_Image','_Truth'), sitk.sitkUInt8)
        truth_array = sitk.GetArrayFromImage(truth)
        volume = truth_array[truth_array == 1].shape[0] * np.prod(truth.GetSpacing()) / 1000
        prediction = sitk.ReadImage(file.replace('_Image','_Prediction'))
        prediction = resampler.resample_image(prediction, ref_handle=truth)
        overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()

        statistics_image_filter = sitk.StatisticsImageFilter()

        out_dict = {'volume':volume}
        for _, measured_name in enumerate(OverlapMeasures):
            out_dict[measured_name.name] = np.zeros((len(threshold_range), len(seed_range)))
        for _, measured_name in enumerate(SurfaceDistanceMeasures):
            out_dict[measured_name.name] = np.ones((len(threshold_range), len(seed_range)))*999
        out_dict['Sensitivity'] = np.zeros((len(threshold_range), len(seed_range)))
        out_dict['Specificity'] = np.zeros((len(threshold_range), len(seed_range)))
        reference_surface = sitk.LabelContour(truth)
        reference_distance_map = sitk.Abs(sitk.SignedMaurerDistanceMap(truth, squaredDistance=False,
                                                                       useImageSpacing=True))
        statistics_image_filter.Execute(reference_surface)
        fill_binary = Fill_Binary_Holes()
        Connected_Component_Filter = sitk.ConnectedComponentImageFilter()
        Connected_Threshold = sitk.ConnectedThresholdImageFilter()
        Connected_Threshold.SetUpper(2)
        stats = sitk.LabelShapeStatisticsImageFilter()
        staple = sitk.STAPLEImageFilter()
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
                staple.Execute(threshold_pred, truth)
                out_dict['Sensitivity'][i, j] = staple.GetSensitivity()[0]
                out_dict['Specificity'][i, j] = staple.GetSpecificity()[0]
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
                out_dict[SurfaceDistanceMeasures.mean_surface_distance.name][i, j] = np.mean(
                    all_surface_distances)

                out_dict[SurfaceDistanceMeasures.median_surface_distance.name][i, j] = np.median(
                    all_surface_distances)
                out_dict[SurfaceDistanceMeasures.std_surface_distance.name][i, j] = np.std(
                    all_surface_distances)
                out_dict[SurfaceDistanceMeasures.max_surface_distance.name][i, j] = np.max(
                    all_surface_distances)
        save_obj(os.path.join(self.save_path,'{}_out_dict_single_patient.pkl'.format(pat_name)),out_dict)


def worker_def_index_single_disease(A):
    q = A[0]
    base_class = run_index_single_disease()
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


class run_metrics_single_disease(object):
    def __init__(self, save_path):
        self.save_path = save_path

    def process(self, threshold_range, seed_range, file, thread_count=int(cpu_count()*.95-1),
                write_final_prediction=False):
        q = Queue(maxsize=thread_count)
        A = [q, ]
        threads = []
        for worker in range(thread_count):
            t = Thread(target=worker_def_index_single_disease, args=(A,))
            t.start()
            threads.append(t)
        reader = sitk.ImageFileReader()
        reader.LoadPrivateTagsOn()
        pat_name = os.path.split(file)[-1].split('.')[0]
        print(pat_name)
        truth_base = sitk.ReadImage(file.replace('_Image','_Truth'), sitk.sitkUInt8)
        prediction = sitk.ReadImage(file.replace('_Image','_Prediction'))
        prediction = sitk.GetArrayFromImage(prediction)

        Connected_Component_Filter = sitk.ConnectedComponentImageFilter()
        stats = sitk.LabelShapeStatisticsImageFilter()
        connected_truth = Connected_Component_Filter.Execute(truth_base)
        stats.Execute(connected_truth)
        tumor_labels = stats.GetLabels()
        out_dict = {'volume':[stats.GetNumberOfPixels(i)*np.prod(truth_base.GetSpacing())/1000 for i in tumor_labels]}  # In cc
        for _, measured_name in enumerate(OverlapMeasures):
            out_dict[measured_name.name] = np.zeros((len(threshold_range), len(seed_range), len(tumor_labels)))
        for _, measured_name in enumerate(SurfaceDistanceMeasures):
            out_dict[measured_name.name] = np.ones((len(threshold_range), len(seed_range), len(tumor_labels)))*999
        total = len(threshold_range) ** 1 * len(seed_range) ** 1
        checkpoint = 0
        for i, threshold_value in enumerate(threshold_range):
            print('Threshold value {}'.format(threshold_value))
            for j, seed_value in enumerate(seed_range):
                already_done = False
                for k, tumor_label in enumerate(tumor_labels):
                    index = tuple([i, j, k])
                    if out_dict[OverlapMeasures.jaccard.name][index] != 0:
                        already_done = True
                checkpoint += 1
                percentage = checkpoint / total * 100
                if already_done:
                    print("Already done")
                    print('{}% Done'.format(percentage))
                    continue
                item = {'base_index': [i, j],
                        'seed_value': seed_value,
                        'threshold_value': threshold_value,
                        'prediction': copy.deepcopy(prediction),
                        'connected_truth': connected_truth,
                        'out_dict': out_dict,
                        'tumor_labels': tumor_labels,
                        'percentage': percentage,
                        'write_final_prediction': write_final_prediction}
                q.put(item)
        for i in range(thread_count):
            q.put(None)
        for t in threads:
            t.join()
        save_obj(os.path.join(self.save_path,'{}_out_dict_single_disease.pkl'.format(pat_name)),out_dict)


def worker_def_single_patient(A):
    q, save_path = A
    base_class = run_metrics_single_patient(save_path)
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


def combine_patient_pickles(out_path, is_disease=True):
    if is_disease:
        add = 'disease'
    else:
        add = 'patient'
    image_list = [os.path.join(out_path, i) for i in os.listdir(out_path) if i.find('_out_dict_single_{}.pkl'.format(add)) != -1]
    out_dict = {'patient_name': []}
    for file in image_list:
        pat_name = os.path.split(file)[-1].split('_out_dict')[0]
        patient_dict = load_obj(file)
        for key in patient_dict:
            data = patient_dict[key]
            if not is_disease:
                data = [data]
            if key not in out_dict:
                out_dict[key] = data
            else:
                if type(data) is list:
                    out_dict[key] += data
                elif is_disease:
                    out_dict[key] = np.concatenate([out_dict[key], data], axis=-1)
        out_dict['patient_name'] += [pat_name for _ in range(np.asarray(patient_dict[key]).shape[-1])]
    if is_disease:
        for key in out_dict.keys():
            print(key)
            if type(out_dict[key][0]) is not np.ndarray:
                out_dict[key] = np.asarray(out_dict[key])
    return out_dict


def find_best_threshold_seed(threshold_range, seed_range, out_path):
    out_dict = combine_patient_pickles(out_path)
    new_output = {'Volume': np.squeeze(out_dict['volume']), 'patient_name': np.squeeze(out_dict['patient_name']),
                  'dice': [], 'seed_value': [], 'threshold_value': []}
    data = np.round(out_dict['dice'], 4)
    values = np.column_stack(np.unravel_index(data.reshape(data.shape[-1],-1).argmax(axis=1), data[...,0].shape))
    threshold_indexes = values[:, 0]
    seed_indexes = values[:, 1]
    new_output['threshold_value'] = threshold_range[threshold_indexes]
    new_output['seed_value'] = seed_range[seed_indexes]
    new_output['dice'] = np.max(data,axis=(0,1))
    df = pd.DataFrame(new_output)
    df.to_excel(os.path.join(out_path,'pattern_draw.xlsx'), index=0)
    threshold_names = ['Threshold_{}'.format(i) for i in threshold_range]
    seed_names = ['Seed_{}'.format(i) for i in seed_range]
    volume = out_dict['volume']
    for key in out_dict.keys():
        if key == 'volume' or key == 'patient_name' or key.find('distance') != -1:
            continue
        with pd.ExcelWriter(os.path.join(out_path,'{}.xlsx'.format(key))) as writer:
            lower_volume = 0
            for volume_threshold in [1, 5, 20, 9999]:
                mask_upper = volume > lower_volume
                mask_lower = volume <= volume_threshold
                mask = mask_upper * mask_lower
                data = np.median(out_dict[key][..., mask], axis=-1)
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
                df.to_excel(writer, sheet_name='{} <= x < {}cc'.format(lower_volume, volume_threshold))
                lower_volume = volume_threshold
    return None


def create_metric_chart(path=r'H:\Liver_Disease_Ablation\Predictions\Validation', out_path=os.path.join('.', 'Threshold'),
                        threshold_range=[0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95],
                        seed_range=[0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95],
                        thread_count=int(cpu_count()*.95-1), re_write=False, write_final_prediction=False,
                        single_disease=True):
    image_list = [os.path.join(path,i) for i in os.listdir(path) if i.find('_Image') != -1]
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    if single_disease:
        single_disease_runner = run_metrics_single_disease(out_path)
        for file in image_list:
            pat_name = os.path.split(file)[-1].split('.')[0]
            if os.path.exists(os.path.join(out_path, '{}_out_dict_single_disease.pkl'.format(pat_name))) and not re_write:
                continue
            item = {'threshold_range': threshold_range, 'seed_range': seed_range, 'file': file,
                    'thread_count': thread_count, 'write_final_prediction': write_final_prediction}
            single_disease_runner.process(**item)
    else:
        q = Queue(maxsize=thread_count)
        A = [q, out_path]
        threads = []
        for worker in range(thread_count):
            t = Thread(target=worker_def_single_patient, args=(A,))
            t.start()
            threads.append(t)
        for file in image_list:
            pat_name = os.path.split(file)[-1].split('.')[0]
            if os.path.exists(
                    os.path.join(out_path, '{}_out_dict_single_patient.pkl'.format(pat_name))) and not re_write:
                continue
            item = {'threshold_range': threshold_range, 'seed_range': seed_range,
                    'write_final_prediction': write_final_prediction, 'file': file}
            q.put(item)
        for i in range(thread_count):
            q.put(None)
        for t in threads:
            t.join()

    if not write_final_prediction:
        find_best_threshold_seed(threshold_range=threshold_range, seed_range=seed_range, out_path=out_path)
        return None
    out_dict = combine_patient_pickles(out_path, is_disease=single_disease)
    for key in out_dict.keys():
        out_dict[key] = np.squeeze(out_dict[key])
    df = pd.DataFrame(out_dict)
    df.to_excel(os.path.join(out_path,'Final_Prediction_93.xlsx'), index=0)


if __name__ == '__main__':
    pass
