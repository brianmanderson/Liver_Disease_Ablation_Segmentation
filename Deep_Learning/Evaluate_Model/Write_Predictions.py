__author__ = 'Brian M Anderson'
# Created on 3/2/2020
from tensorflow.python.keras.models import *
import tensorflow.compat.v1 as tf
from Base_Deeplearning_Code.Keras_Utils.Keras_Utilities import dice_coef_3D, np
import os, time
from Return_Train_Validation_Generators import return_generators, sitk, plot_scroll_Image


def create_prediction_files(is_test=False, path_ext = ''):
    path_extension = 'Single_Images3D' + path_ext
    cube_size = (30,300,300)
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
        pred_output_path = os.path.join('D:\Liver_Disease_Ablation\Predictions{}'.format(path_ext),ext)
        if not os.path.exists(pred_output_path):
            os.makedirs(pred_output_path)
        for i in range(len(eval_generator)):
            image_path = eval_generator.image_list[i][0]
            image_name = ''.join(image_path.split('\\')[-1].split('_')[:-2])
            print(image_name)
            if os.path.exists(os.path.join(pred_output_path, '{}_Image.nii.gz'.format(image_name))):
                continue
            elif model_val is None:
                model_val = load_model(r'D:\Liver_Disease_Ablation\weights-improvement-best_10_90.hdf5',
                                       custom_objects={'dice_coef_3D': dice_coef_3D})
            x,y = eval_generator.__getitem__(i)
            sitk_image = sitk.ReadImage(image_path)
            spacing = sitk_image.GetSpacing() + (1,)
            pred = model_val.predict(x)

            truth = sitk.GetImageFromArray(np.squeeze(y[...,1]))
            truth.SetSpacing(spacing)
            sitk.WriteImage(truth, os.path.join(pred_output_path, '{}_Truth.nii.gz'.format(image_name)))

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

