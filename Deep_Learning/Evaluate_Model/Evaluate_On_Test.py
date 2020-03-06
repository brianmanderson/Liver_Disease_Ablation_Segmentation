__author__ = 'Brian M Anderson'
# Created on 3/4/2020

import os, copy
import SimpleITK as sitk
from skimage import morphology
import numpy as np
from enum import Enum
from Base_Deeplearning_Code.Plot_And_Scroll_Images.Plot_Scroll_Images import plot_scroll_Image
import pandas as pd


def remove_non_liver(pred, min_volume=0.0, max_area=np.inf, min_area=0.0, spacing=None):
    '''
    :param annotations: An annotation of shape [Z_images, rows, columns]
    :param threshold: Threshold of probability from 0.0 to 1.0
    :param max_volume: Max volume of structure allowed
    :param min_volume: Minimum volume of structure allowed, in ccs
    :param max_area: Max volume of structure allowed
    :param min_area: Minimum volume of structure allowed
    :param do_3D: Do a 3D removal of structures, only take largest connected structure
    :param do_2D: Do a 2D removal of structures, only take largest connected structure
    :param spacing: Spacing of elements, in form of [z_spacing, row_spacing, column_spacing]
    :return: Masked annotation
    '''
    min_volume = min_volume * 1000  # cm3 to mm3
    min_area = min_area * 100
    max_area = max_area * 100
    Connected_Component_Filter = sitk.ConnectedComponentImageFilter()
    RelabelComponent = sitk.RelabelComponentImageFilter()
    BinaryThreshold = sitk.BinaryThresholdImageFilter()
    BinaryThreshold.SetLowerThreshold(0.001)
    if min_volume != 0:
        label_image = Connected_Component_Filter.Execute(BinaryThreshold.Execute(sitk.GetImageFromArray(pred.astype('float32'))))
        RelabelComponent.SetMinimumObjectSize(int(min_volume / np.prod(spacing)))
        label_image = RelabelComponent.Execute(label_image)
        pred = sitk.GetArrayFromImage(label_image)
        pred[pred > 0] = 1
        pred[pred < 1] = 0
    if min_area != 0 or max_area != np.inf:
        slice_indexes = np.where(np.sum(pred, axis=(1, 2)) > 0)
        if slice_indexes:
            slice_spacing = np.prod(spacing[:-1])
            for slice_index in slice_indexes[0]:
                labels = morphology.label(pred[slice_index], connectivity=1)
                for i in range(1, labels.max() + 1):
                    new_area = labels[labels == i].shape[0]
                    temp_area = slice_spacing * new_area
                    if temp_area > max_area:
                        labels[labels == i] = 0
                        continue
                    elif temp_area < min_area:
                        labels[labels == i] = 0
                        continue
                labels[labels > 0] = 1
                pred[slice_index] = labels
    if min_volume != 0:
        label_image = Connected_Component_Filter.Execute(BinaryThreshold.Execute(sitk.GetImageFromArray(pred.astype('float32'))))
        RelabelComponent.SetMinimumObjectSize(int(min_volume / np.prod(spacing)))
        label_image = RelabelComponent.Execute(label_image)
        pred = sitk.GetArrayFromImage(label_image)
        pred[pred > 0] = 1
        pred[pred < 1] = 0
    return pred


class OverlapMeasures(Enum):
    jaccard, dice, volume_similarity, false_negative, false_positive = range(5)

class SurfaceDistanceMeasures(Enum):
    hausdorff_distance, mean_surface_distance, median_surface_distance, std_surface_distance, max_surface_distance = range(5)


def create_metric_chart(path = r'D:\Liver_Disease_Ablation\Predictions_None\Test', threshold=0.55):
    image_list = [os.path.join(path,i) for i in os.listdir(path) if i.find('_Image') != -1]
    out_dict = {}
    for name, _ in OverlapMeasures.__members__.items():
        out_dict[name] = {'Patient_ID':['']}
    for name, _ in SurfaceDistanceMeasures.__members__.items():
        out_dict[name] = {'Patient_ID':['']}
    metric = 'Test_Output'
    out_path = os.path.join('.',metric)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    for name in out_dict.keys():
        for i in [0,1]:
            out_dict[name]['{}_{}'.format(metric,i)] = [i]
    for file in image_list:
        pat_name = file.split('\\')[-1].split('_')[0]
        print(pat_name)
        for name in out_dict.keys():
            out_dict[name]['Patient_ID'].append(pat_name)
        truth = sitk.ReadImage(file.replace('_Image','_Truth'), sitk.sitkUInt8)
        truth_array = sitk.GetArrayFromImage(truth)
        volume = truth_array[truth_array==1].shape[0] * np.prod(truth.GetSpacing())/(1000)
        for name in out_dict.keys():
            out_dict[name]['{}_{}'.format(metric,1)].append(volume)
        prediction = sitk.ReadImage(file.replace('_Image','_Prediction'))

        overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()

        hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()

        statistics_image_filter = sitk.StatisticsImageFilter()

        overlap_results = np.zeros((len([0]), len(OverlapMeasures.__members__.items())))
        surface_distance_results = np.zeros((len([0]), len(SurfaceDistanceMeasures.__members__.items())))

        reference_distance_map = sitk.Abs(sitk.SignedMaurerDistanceMap(truth, squaredDistance=False,
                                                                       useImageSpacing=True))
        reference_surface = sitk.LabelContour(truth)

        statistics_image_filter.Execute(reference_surface)
        num_reference_surface_pixels = int(statistics_image_filter.GetSum())

        # for i, threshold in enumerate(threshold_range):
        Binary_Fill_Filter = sitk.BinaryFillholeImageFilter()
        Binary_Fill_Filter.SetFullyConnected(True)

        Binary_Threshold = sitk.BinaryThresholdImageFilter()
        i = 0
        Binary_Threshold.SetLowerThreshold(threshold)
        threshold_pred = Binary_Threshold.Execute(prediction)
        output_size = threshold_pred.GetSize()
        output_array = np.zeros(np.flip(output_size))
        for slice_index in range(output_size[-1]):
            filled = Binary_Fill_Filter.Execute(threshold_pred[:, :, slice_index])
            output_array[slice_index] = sitk.GetArrayFromImage(filled)
        try:
            output_array = remove_non_liver(output_array,min_volume=1, min_area=0, spacing=truth.GetSpacing())
            threshold_pred = sitk.GetImageFromArray(output_array.astype('float32'))
            threshold_pred.SetSpacing(prediction.GetSpacing())
            threshold_pred = Binary_Threshold.Execute(threshold_pred)
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
        except:
            xxx = 1
        for _, measured_name in enumerate(OverlapMeasures):
            out_dict[measured_name.name]['{}_{}'.format(metric, 0)].append(
                overlap_results[i, measured_name.value])
            print('{} is {}'.format(measured_name.name, overlap_results[i, measured_name.value]))
        for _, measured_name in enumerate(SurfaceDistanceMeasures):
            out_dict[measured_name.name]['{}_{}'.format(metric, 0)].append(
                surface_distance_results[i, measured_name.value])
        for measured_name in out_dict.keys():
            df = pd.DataFrame(out_dict[measured_name])
            df.to_excel(os.path.join(out_path,'Out_Data_{}.xlsx'.format(measured_name)),index=0)
    return None


if __name__ == '__main__':
    create_metric_chart()
