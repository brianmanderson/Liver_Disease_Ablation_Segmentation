__author__ = 'Brian M Anderson'
# Created on 2/4/2021
import numpy as np
import os
import SimpleITK as sitk
from Deep_Learning.Base_Deeplearning_Code.Dicom_RT_and_Images_to_Mask.src.DicomRTTool import DicomReaderWriter, \
    plot_scroll_Image


class Threshold_and_Expand(object):
    def __init__(self, seed_threshold_value=None, lower_threshold_value=None):
        self.seed_threshold_value = seed_threshold_value
        self.Connected_Component_Filter = sitk.ConnectedComponentImageFilter()
        self.RelabelComponent = sitk.RelabelComponentImageFilter()
        self.Connected_Threshold = sitk.ConnectedThresholdImageFilter()
        self.stats = sitk.LabelShapeStatisticsImageFilter()
        self.lower_threshold_value = lower_threshold_value
        self.Connected_Threshold.SetUpper(2)

    def post_process(self, images, pred, ground_truth=None):
        for i in range(1, pred.shape[-1]):
            temp_pred = pred[..., i]
            output = np.zeros(temp_pred.shape)
            expanded = False
            if len(temp_pred.shape) == 4:
                temp_pred = temp_pred[0]
                expanded = True
            prediction = sitk.GetImageFromArray(temp_pred)
            if type(self.seed_threshold_value) is not list:
                seed_threshold = self.seed_threshold_value
            else:
                seed_threshold = self.seed_threshold_value[i - 1]
            if type(self.lower_threshold_value) is not list:
                lower_threshold = self.lower_threshold_value
            else:
                lower_threshold = self.lower_threshold_value[i - 1]
            overlap = temp_pred > seed_threshold
            if np.max(overlap) > 0:
                seeds = np.transpose(np.asarray(np.where(overlap > 0)))[..., ::-1]
                seeds = [[int(i) for i in j] for j in seeds]
                self.Connected_Threshold.SetLower(lower_threshold)
                self.Connected_Threshold.SetSeedList(seeds)
                output = sitk.GetArrayFromImage(self.Connected_Threshold.Execute(prediction))
                if expanded:
                    output = output[None, ...]
            pred[..., i] = output
        return images, pred, ground_truth


def write_new_predictions(path=r'H:\AutoModels\Liver_Disease\Input_3',
                          niftii_path=r'H:\Liver_Disease_Ablation\3Dircadb1\Niftii_Files', seed=0.01):
    threshold_expand = Threshold_and_Expand(seed_threshold_value=seed, lower_threshold_value=seed)
    reader_writer = DicomReaderWriter()
    for patient in os.listdir(path):
        if patient in ['Patient_5', 'Patient_11', 'Patient_12', 'Patient_18', 'Patient_20']:  # Exclusion criteria
            continue
        image_path = os.path.join(niftii_path,
                                  '{}_NewPrediction_{}.nii'.format(patient, seed))
        if os.path.exists(image_path):
            continue
        reader_writer.walk_through_folders(input_path=os.path.join(path, patient))
        reader_writer.get_images()
        prediction_array = np.load(os.path.join(path, patient, 'Raw_Pred.npy'))
        _, threshold_pred, _ = threshold_expand.post_process(images=None, pred=prediction_array, ground_truth=None)
        pred_image = sitk.GetImageFromArray(threshold_pred)
        pred_image.SetOrigin(reader_writer.dicom_handle.GetOrigin())
        pred_image.SetSpacing(reader_writer.dicom_handle.GetSpacing())
        pred_image.SetDirection(reader_writer.dicom_handle.GetDirection())
        sitk.WriteImage(pred_image, image_path)
        reader_writer.__reset__()
    return None


if __name__ == '__main__':
    pass
