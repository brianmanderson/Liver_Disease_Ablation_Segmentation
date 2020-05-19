__author__ = 'Brian M Anderson'
# Created on 3/2/2020
from tensorflow.python.keras.models import *
import numpy as np
import os
import SimpleITK as sitk
from Return_Train_Validation_Generators_TF2 import return_generators, plot_scroll_Image


def create_prediction_files(is_test=False, path_ext = '', desc='', model_path='weights-improvement-best_FWHM.hdf5'):
    reader = sitk.ImageFileReader()
    reader.LoadPrivateTagsOn()
    base_path, morfeus_drive, _, eval_generator = return_generators(is_test=is_test, wanted_keys={'inputs':['image_path','image','mask'],'outputs':['annotation']},cache=False)
    model_val = None
    ext = 'Validation'
    if is_test:
        ext = 'Test'
    ext += desc
    pred_output_path = os.path.join('D:\Liver_Disease_Ablation\Predictions{}'.format(path_ext),ext)
    if not os.path.exists(pred_output_path):
        os.makedirs(pred_output_path)
    gen = eval_generator.data_set.as_numpy_iterator()
    for i in range(len(eval_generator)):
        x, y = next(gen)
        if is_test:
            image_name = os.path.split(x[0][0].decode())[-1]
        else:
            image_name = 'Image_{}'.format(i)
        x = x[1:]
        y = y[0]
        print(image_name)
        if os.path.exists(os.path.join(pred_output_path, '{}_Image.nii.gz'.format(image_name))):
            continue
        elif model_val is None:
            model_val = load_model(model_path, compile=False)
        step = 200
        pull = 160
        gap = (step - pull) // 2
        shift = pull
        if x[0].shape[1] > step:
            pred = np.zeros(x[0][0].shape[:-1] + (2,))
            start = 0
            while start < x[0].shape[1]:
                pred_cube = model_val.predict([x[0][:,start:start+step,...],x[1][:,start:start+step,...],x[2][:,start:start+step,...]])
                start_gap = gap
                stop_gap = gap
                if start == 0:
                    start_gap = 0
                elif start + shift >= x[0].shape[1]:
                    stop_gap = 0
                if stop_gap != 0:
                    pred_cube = pred_cube[:, start_gap:-stop_gap, ...]
                else:
                    pred_cube = pred_cube[:, start_gap:, ...]
                pred[:,start+start_gap:start + start_gap + pred_cube.shape[1],...] = pred_cube
                start += shift
        else:
            pred = model_val.predict(x)

        truth = sitk.GetImageFromArray(np.squeeze(y).astype('float32'))
        sitk.WriteImage(truth, os.path.join(pred_output_path, '{}_Truth.nii.gz'.format(image_name)))

        prediction = sitk.GetImageFromArray(np.squeeze(pred[...,1]).astype('float32'))
        prediction.SetOrigin(truth.GetOrigin())
        prediction.SetDirection(truth.GetDirection())
        sitk.WriteImage(prediction, os.path.join(pred_output_path, '{}_Prediction.nii.gz'.format(image_name)))

        image = sitk.GetImageFromArray(np.squeeze(x[0]).astype('float32'))
        image.SetOrigin(truth.GetOrigin())
        image.SetDirection(truth.GetDirection())
        sitk.WriteImage(image, os.path.join(pred_output_path, '{}_Image.nii.gz'.format(image_name)))
    K.clear_session()
    return None


if __name__ == '__main__':
    pass

