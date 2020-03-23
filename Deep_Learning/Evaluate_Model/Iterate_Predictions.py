__author__ = 'Brian M Anderson'
# Created on 3/2/2020
import os, copy
import SimpleITK as sitk
from skimage import morphology
import numpy as np
from enum import Enum
from Base_Deeplearning_Code.Plot_And_Scroll_Images.Plot_Scroll_Images import plot_scroll_Image
import pandas as pd


class OverlapMeasures(Enum):
    jaccard, dice, volume_similarity, false_negative, false_positive = range(5)


class Minimum_Volume_and_Area_Prediction(object):
    '''
    This should come after prediction thresholding
    '''
    def __init__(self, min_volume=0.0):
        '''
        :param min_volume: Minimum volume of structure allowed, in cm3
        :param min_area: Minimum area of structure allowed, in cm2
        :param max_area: Max area of structure allowed, in cm2
        :return: Masked annotation
        '''
        self.min_volume = min_volume * 1000 # cm3 to mm3
        self.Connected_Component_Filter = sitk.ConnectedComponentImageFilter()
        self.RelabelComponent = sitk.RelabelComponentImageFilter()

    def process(self, threshold_image):
        if self.min_volume != 0:
            label_image = self.Connected_Component_Filter.Execute(threshold_image)
            self.RelabelComponent.SetMinimumObjectSize(int(self.min_volume/np.prod(threshold_image.GetSpacing())))
            label_image = self.RelabelComponent.Execute(label_image)
            threshold_image = sitk.BinaryThreshold(sitk.Cast(label_image,sitk.sitkFloat32),
                                                   lowerThreshold=0.1,upperThreshold=200)
        return threshold_image


class SurfaceDistanceMeasures(Enum):
    hausdorff_distance, mean_surface_distance, median_surface_distance, std_surface_distance, max_surface_distance = range(5)


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


def create_metric_chart(path = r'D:\Liver_Disease_Ablation\Predictions_1mm\Validation',
                        metric_range = [0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95],
                        desc='',out_path=os.path.join('.','Threshold')):
    image_list = [os.path.join(path,i) for i in os.listdir(path) if i.find('_Image') != -1]
    out_dict = {}
    for name, _ in OverlapMeasures.__members__.items():
        out_dict[name] = {'Patient_ID':['']}
    for name, _ in SurfaceDistanceMeasures.__members__.items():
        out_dict[name] = {'Patient_ID':['']}
    metric = 'Threshold'
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    for name in out_dict.keys():
        for i in metric_range:
            out_dict[name]['{}_{}'.format(metric,i)] = [i]
        out_dict[name]['Volume'] = [0]


    overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()

    hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()

    statistics_image_filter = sitk.StatisticsImageFilter()

    fill_binary = Fill_Binary_Holes()
    min_volume = Minimum_Volume_and_Area_Prediction(5)
    for file in image_list:
        pat_name = file.split('\\')[-1].split('_')[0]
        print(pat_name)
        for name in out_dict.keys():
            out_dict[name]['Patient_ID'].append(pat_name)
        truth = sitk.ReadImage(file.replace('_Image','_Truth'), sitk.sitkUInt8)
        prediction = sitk.ReadImage(file.replace('_Image','_Prediction'))
        truth_array = sitk.GetArrayFromImage(truth)
        volume = truth_array[truth_array == 1].shape[0] * np.prod(truth.GetSpacing()) / 1000
        print(volume)
        for name in out_dict.keys():
            out_dict[name]['Volume'].append(volume)
        overlap_results = np.zeros((len(metric_range), len(OverlapMeasures.__members__.items())))
        surface_distance_results = np.zeros((len(metric_range), len(SurfaceDistanceMeasures.__members__.items())))

        reference_distance_map = sitk.Abs(sitk.SignedMaurerDistanceMap(truth, squaredDistance=False,
                                                                       useImageSpacing=True))
        reference_surface = sitk.LabelContour(truth)

        statistics_image_filter.Execute(reference_surface)
        num_reference_surface_pixels = int(statistics_image_filter.GetSum())

        # for i, threshold in enumerate(threshold_range):
        print(metric)
        for i, metric_value in enumerate(metric_range):
            threshold_and_expand = Threshold_and_Expand(seed_threshold_value=0.95, lower_threshold_value=.2)
            print(metric_value)
            threshold_pred = threshold_and_expand.process(prediction)
            # threshold_pred = fill_binary.process(threshold_pred)
            threshold_pred = min_volume.process(threshold_pred)
            overlap_measures_filter.Execute(truth, threshold_pred)
            try:
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
                out_dict[measured_name.name]['{}_{}'.format(metric, metric_value)].append(
                    overlap_results[i, measured_name.value])
            for _, measured_name in enumerate(SurfaceDistanceMeasures):
                out_dict[measured_name.name]['{}_{}'.format(metric, metric_value)].append(
                    surface_distance_results[i, measured_name.value])
        for measured_name in out_dict.keys():
            df = pd.DataFrame(out_dict[measured_name])
            df.to_excel(os.path.join(out_path,'Out_Data_{}_{}.xlsx'.format(desc, measured_name)),index=0)
    return None


if __name__ == '__main__':
    pass
