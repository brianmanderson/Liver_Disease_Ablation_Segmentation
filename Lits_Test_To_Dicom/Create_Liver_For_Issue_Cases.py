__author__ = 'Brian M Anderson'
# Created on 9/23/2020
'''
I noticed that some of the nifti to dicom conversions were failing... so no we'll have to make
some of the structures here and work on them later
'''


from Image_Processing import *
from Bilinear_Dsc import BilinearUpsampling
from DicomRTTool import plot_scroll_Image


model_loader = {'image_processors': [
                Normalize_Images(mean_val=0, std_val=1, lower_threshold=-100, upper_threshold=300, max_val=255),
                Expand_Dimension(axis=-1), Repeat_Channel(num_repeats=3, axis=-1),
                Ensure_Image_Proportions(image_rows=512, image_cols=512),
                VGG_Normalize()],
            'prediction_processors': [Threshold_Prediction(threshold=0.5, single_structure=True,
                                                           is_liver=True)]}
model_load_path = r'K:\Morfeus\Auto_Contour_Sites\Models'
model_path = os.path.join(model_load_path, 'Liver', 'weights-improvement-512_v3_model_xception-36.hdf5')
graph = tf.compat.v1.Graph()
with graph.as_default():
    gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
    session = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(
        gpu_options=gpu_options, log_device_placement=False))
    with session.as_default():
        tf.compat.v1.keras.backend.set_session(session)
        model = tf.keras.models.load_model(model_path, custom_objects={'BilinearUpsampling':BilinearUpsampling,
                                                                       'dice_coef_3D':dice_coef_3D,'loss':None},
                                           compile=False)
        for pat_id in ['57', '59']:
            path = os.path.join(r'H:\Liver_Disease_Ablation\LiTs_Test\Nifti', 'test-volume-{}.nii'.format(pat_id))
            xxx = 1
            image_handle = sitk.ReadImage(path)
            images = sitk.GetArrayFromImage(image_handle)
            if pat_id == '57':
                images = images[:, ::-1]

            for processor in model_loader['image_processors']:
                print('Performing pre process {}'.format(processor))
                images, ground_truth = processor.pre_process(images, None)

            pred = model.predict(images)

            for processor in model_loader['image_processors'][::-1]:  # In reverse now
                print('Performing post process {}'.format(processor))
                images, pred, ground_truth = processor.post_process(images, pred, None)
            for processor in model_loader['prediction_processors']:
                print('Performing prediction process {}'.format(processor))
                images, pred, ground_truth = processor.post_process(images, pred, None)
            pred = pred[..., -1]
            if pat_id == '57':
                pred = pred[:, ::-1]
            pred_handle = sitk.GetImageFromArray(pred)
            pred_handle.SetSpacing(image_handle.GetSpacing())
            pred_handle.SetOrigin(image_handle.GetOrigin())
            pred_handle.SetDirection(image_handle.GetDirection())
            sitk.WriteImage(pred_handle, os.path.join('.', 'pred.nii.gz'))
            break