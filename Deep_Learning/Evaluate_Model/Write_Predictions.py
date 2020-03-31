__author__ = 'Brian M Anderson'
# Created on 3/2/2020
from tensorflow.python.keras.models import *
import tensorflow.compat.v1 as tf
from Base_Deeplearning_Code.Keras_Utils.Keras_Utilities import dice_coef_3D, np
import os, time
from Return_Train_Validation_Generators import return_generators, sitk, plot_scroll_Image


def create_prediction_files(is_test=False, path_ext = '', desc='', model_name='weights-improvement-best_FWHM.hdf5'):
    path_extension = 'Single_Images3D' + path_ext
    cube_size = (30,300,300)
    reader = sitk.ImageFileReader()
    reader.LoadPrivateTagsOn()
    # cube_size = None
    num_patients = 1
    base_path, morfeus_drive, _, eval_generator = return_generators(liver_norm=True,
                                                                    cube_size=cube_size,
                                                                    path_extension=path_extension,
                                                                    num_patients=num_patients,
                                                                    return_test=is_test)
    gpu = 0
    with tf.device('/gpu:' + str(gpu)):
        gpu_options = tf.GPUOptions(allow_growth=True) # maybe should just allocate whole gpu..
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        tf.keras.backend.set_session(sess)
        model_val = None
        ext = 'Validation'
        if is_test:
            ext = 'Test'
        ext += desc
        pred_output_path = os.path.join('D:\Liver_Disease_Ablation\Predictions{}'.format(path_ext),ext)
        if not os.path.exists(pred_output_path):
            os.makedirs(pred_output_path)
        for i in range(len(eval_generator)):
            image_path = eval_generator.image_list[i][0]
            image_name = ''.join(image_path.split('\\')[-1].split('_')[:-2])
            load_path_index = image_path.index('Single_Images')
            load_path = image_path[:load_path_index]
            print(image_name)
            if os.path.exists(os.path.join(pred_output_path, '{}_Image.nii.gz'.format(image_name))):
                continue
            elif model_val is None:
                model_val = load_model(r'D:\Liver_Disease_Ablation\{}'.format(model_name),
                                       custom_objects={'dice_coef_3D': dice_coef_3D})
            x,y = eval_generator.__getitem__(i)
            whole_patient_file_name = 'Overall_Data_{}'.format(image_path.split('\\')[-1].replace('_{}_image'.format(image_path.split('_')[-2]),''))
            reader.SetFileName(os.path.join(load_path,whole_patient_file_name))
            reader.ReadImageInformation()
            spacing = reader.GetSpacing()
            step = 200
            pull = 160
            gap = (step - pull) // 2
            shift = pull
            if x[0].shape[1] > step:
                pred = np.zeros(x[0].shape[:-1] + (2,))
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

            truth = sitk.GetImageFromArray(np.squeeze(y[...,1]))
            truth.SetSpacing(spacing)
            sitk.WriteImage(truth, os.path.join(pred_output_path, '{}_Truth.nii.gz'.format(image_name)))

            liver = sitk.GetImageFromArray(np.squeeze(x[1][...,0]))
            liver.SetSpacing(spacing)
            sitk.WriteImage(liver, os.path.join(pred_output_path, '{}_Liver.nii.gz'.format(image_name)))

            prediction = sitk.GetImageFromArray(np.squeeze(pred[...,1]))
            prediction.SetSpacing(spacing)
            prediction.SetOrigin(truth.GetOrigin())
            prediction.SetDirection(truth.GetDirection())
            sitk.WriteImage(prediction, os.path.join(pred_output_path, '{}_Prediction.nii.gz'.format(image_name)))

            image = sitk.GetImageFromArray(np.squeeze(x[0]))
            image.SetSpacing(spacing)
            image.SetOrigin(truth.GetOrigin())
            image.SetDirection(truth.GetDirection())
            sitk.WriteImage(image, os.path.join(pred_output_path, '{}_Image.nii.gz'.format(image_name)))
    K.clear_session()
    return None


if __name__ == '__main__':
    pass

