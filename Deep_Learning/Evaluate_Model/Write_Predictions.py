__author__ = 'Brian M Anderson'
# Created on 3/2/2020
from tensorflow.python.keras.models import *
import numpy as np
import os, time
import SimpleITK as sitk
from Base_Deeplearning_Code.Data_Generators.Image_Processors_Module.Resample_Class.Resample_Class import Resample_Class_Object
from Return_Train_Validation_Generators_TF2 import return_generators, plot_scroll_Image


def create_prediction_files(is_test=False, path_ext = '', desc='', model_path='weights-improvement-best_FWHM.hdf5',
                            cache=True, validation_path=None, rewrite=False,
                            out_path='H:\Liver_Disease_Ablation\Predictions_93{}'):
    resampler = Resample_Class_Object()
    reader = sitk.ImageFileReader()
    reader.LoadPrivateTagsOn()
    base_path, morfeus_drive, _, eval_generator = return_generators(is_test=is_test, cache_add='Prediction',
                                                                    flip=False, change_background=True, batch_size=8,
                                                                    threshold=True, threshold_val=10,cache=cache, add='_32',
                                                                    wanted_keys={'inputs':
                                                                                     ['image_path', 'image', 'mask'],
                                                                                 'outputs': ['annotation']},
                                                                    evaluation=False, validation_name='Validation_whole',
                                                                    validation_path=validation_path)
    model_val = None
    ext = 'Validation'
    if is_test:
        ext = 'Test'
    ext += desc
    pred_output_path = os.path.join(out_path.format(path_ext),ext)
    total_time = []
    if not os.path.exists(pred_output_path):
        os.makedirs(pred_output_path)
    gen = eval_generator.data_set.as_numpy_iterator()
    for i in range(len(eval_generator)):
        x, y = next(gen)
        image_path = x[0][0].decode()
        if not os.path.exists(image_path):
            image_path = image_path.replace('D:', 'H:') # Moved data files...
        image_name = os.path.split(image_path)[-1].split('.nii.gz')[0]
        print(image_path)
        if os.path.exists(os.path.join(pred_output_path, '{}_Image.nii.gz'.format(image_name))) and not rewrite:
            continue
        elif model_val is None:
            model_val = load_model(model_path, compile=False)
        start_time = time.time()
        x = x[1:]
        y = y[0]
        reader.SetFileName(image_path)
        reader.Execute()
        padded = False
        spacing = reader.GetSpacing()
        image_handle = sitk.GetImageFromArray(np.squeeze(x[0]).astype('float32'))
        mask_handle = sitk.GetImageFromArray(np.squeeze(x[1]).astype('float32'))
        for handle in [image_handle, mask_handle]:
            handle.SetSpacing(spacing)
            handle.SetOrigin(reader.GetOrigin())
            handle.SetDirection(reader.GetDirection())
        # input_spacing = image_handle.GetSpacing()
        resampled_image_handle = resampler.resample_image(image_handle,
                                                          output_spacing=(spacing[0], spacing[1], 1.))
        resampled_mask_handle = resampler.resample_image(mask_handle,
                                                         output_spacing=(spacing[0], spacing[1], 1.))
        image = sitk.GetArrayFromImage(resampled_image_handle)[None,...,None]
        mask = sitk.GetArrayFromImage(resampled_mask_handle)[None,...,None]
        mask[mask > 0.5] = 1
        mask = mask.astype('int')
        image[mask == 0] = 0
        if image.shape[1] % 2 != 0:
            padded = True
            image = np.pad(image,[[0,0],[1,0],[0,0],[0,0],[0,0]])
            mask = np.pad(mask,[[0,0],[1,0],[0,0],[0,0],[0,0]])
        x = [image, mask]

        step = 160//2
        shift = 120//2
        gap = 20//2
        if x[0].shape[1] > step:
            pred = np.zeros(x[0][0].shape[:-1] + (2,))
            start = 0
            while start < x[0].shape[1]:
                pred_cube = model_val.predict([x[0][:,start:start+step,...],x[1][:,start:start+step,...]])
                start_gap = gap
                stop_gap = gap
                if start == 0:
                    start_gap = 0
                elif start + step >= x[0].shape[1]:
                    stop_gap = 0
                if stop_gap != 0:
                    pred_cube = pred_cube[:, start_gap:-stop_gap, ...]
                else:
                    pred_cube = pred_cube[:, start_gap:, ...]
                pred[start+start_gap:start + start_gap + pred_cube.shape[1],...] = pred_cube[0,...]
                start += shift
        else:
            pred = model_val.predict(x)
        stop_time = time.time()
        total_time.append(stop_time-start_time)
        print(np.mean(total_time))
        print(np.std(total_time))
        pred = np.squeeze(pred[...,1])
        if padded:
            pred = pred[1:]

        pred_handle = sitk.GetImageFromArray(pred)
        pred_handle.SetSpacing(resampled_image_handle.GetSpacing())
        pred_handle = resampler.resample_image(pred_handle, image_handle)
        pred = sitk.GetArrayFromImage(pred_handle)

        truth = sitk.GetImageFromArray(np.squeeze(y).astype('float32'))
        truth.SetDirection(image_handle.GetDirection())
        truth.SetOrigin(image_handle.GetOrigin())
        truth.SetDirection(image_handle.GetDirection())
        truth.SetSpacing(reader.GetSpacing())
        sitk.WriteImage(truth, os.path.join(pred_output_path, '{}_Truth.nii.gz'.format(image_name)))

        # mask = sitk.GetImageFromArray(np.squeeze(mask_base).astype('float32'))
        # mask.SetDirection(image_handle.GetDirection())
        # mask.SetOrigin(image_handle.GetOrigin())
        # mask.SetDirection(image_handle.GetDirection())
        # mask.SetSpacing(reader.GetSpacing())
        # sitk.WriteImage(mask, os.path.join(pred_output_path, '{}_Mask.nii.gz'.format(image_name)))

        prediction = sitk.GetImageFromArray(np.squeeze(pred).astype('float32'))
        prediction.SetOrigin(image_handle.GetOrigin())
        prediction.SetDirection(image_handle.GetDirection())
        prediction.SetSpacing(image_handle.GetSpacing())
        sitk.WriteImage(prediction, os.path.join(pred_output_path, '{}_Prediction.nii.gz'.format(image_name)))

        sitk.WriteImage(image_handle, os.path.join(pred_output_path, '{}_Image.nii.gz'.format(image_name)))
    K.clear_session()
    return None


if __name__ == '__main__':
    pass

