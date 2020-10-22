__author__ = 'Brian M Anderson'
# Created on 9/23/2020
'''
I noticed that some of the nifti to dicom conversions were failing... so no we'll have to make
some of the structures here and work on them later
'''

from Image_Processing import *
from Bilinear_Dsc import BilinearUpsampling
from DicomRTTool import plot_scroll_Image


def predict(images, model):
    x = images
    step = 64
    shift = 16
    gap = 16
    if x[0].shape[1] > step:
        pred = np.zeros(x[0][0].shape[:-1] + (2,))
        start = 0
        while start < x[0].shape[1]:
            print(start)
            image_cube, mask_cube = x[0][:, start:start + step, ...], x[1][:, start:start + step, ...]
            difference = image_cube.shape[1] % 32
            if difference != 0:
                image_cube = np.pad(image_cube, [[0, 0], [difference, 0], [0, 0], [0, 0], [0, 0]])
                mask_cube = np.pad(mask_cube, [[0, 0], [difference, 0], [0, 0], [0, 0], [0, 0]])
            pred_cube = model.predict([image_cube, mask_cube])
            pred_cube = pred_cube[:, difference:, ...]
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
            pred[start + start_gap:start + start_gap + pred_cube.shape[1], ...] = pred_cube[0, ...]
            start += shift
    else:
        image_cube, mask_cube = x[0], x[1]
        difference = image_cube.shape[1] % 32
        if difference != 0:
            image_cube = np.pad(image_cube, [[0, 0], [difference, 0], [0, 0], [0, 0], [0, 0]])
            mask_cube = np.pad(mask_cube, [[0, 0], [difference, 0], [0, 0], [0, 0], [0, 0]])
        pred_cube = model.predict([image_cube, mask_cube])
        pred = pred_cube[:, difference:, ...]
    # pred = self.model.predict(x)
    return pred
model_loader = {'image_processors': [
                Normalize_Images(mean_val=0, std_val=1, lower_threshold=-100, upper_threshold=300, max_val=255),
                Expand_Dimension(axis=-1), Repeat_Channel(num_repeats=3, axis=-1),
                Ensure_Image_Proportions(image_rows=512, image_cols=512),
                VGG_Normalize()],
            'prediction_processors': [Threshold_Prediction(threshold=0.5, single_structure=True,
                                                           is_liver=True)]}

model_loader_disease = {'image_processors': [Box_Images(),
                          Normalize_to_Liver(mirror_max=True),
                          Threshold_Images(lower_bound=-10, upper_bound=10, divide=True),
                          # Resample_Process(desired_output_dim=[None, None, 1.0]),
                          Pad_Images(power_val_z=2 ** 4, power_val_y=2 ** 5, power_val_x=2 ** 5),
                          Expand_Dimension(axis=-1),
                          Expand_Dimension(axis=0),
                          Mask_Prediction_New(),
                          Threshold_and_Expand(seed_threshold_value=0.6, lower_threshold_value=.35)
                      ],
                'prediction_processors': [
                    Fill_Binary_Holes(), Mask_within_Liver(),Minimum_Volume_and_Area_Prediction(min_volume=0.5)
                ]
                        }
models = {'liver': model_loader, 'disease': model_loader_disease}
model_load_path = r'K:\Morfeus\Auto_Contour_Sites\Models'
model_path = os.path.join(model_load_path, 'Liver', 'weights-improvement-512_v3_model_xception-36.hdf5')
path = os.path.join(r'H:\Liver_Disease_Ablation\LiTs_Test\Nifti', 'test-volume-59.nii')
out_path = r'H:\Liver_Disease_Ablation\LiTs_Test\Nifti'
image_handle = sitk.ReadImage(path)
images = sitk.GetArrayFromImage(image_handle)
images = np.transpose(images, [1, 0, 2])  # rows, cols, z
images = np.stack([cv2.resize(images[i], (512, 512)) for i in range(512)], axis=0)
key = 'liver'
ground_truth = None
if os.path.exists(os.path.join(out_path, 'test-volume-59-liver.nii')):
    print('Predicting disease')
    model_path = os.path.join(model_load_path, 'Liver_Disease_Ablation', 'Model_42')
    key = 'disease'
    ground_truth = sitk.ReadImage(os.path.join(out_path, 'test-volume-59-liver.nii'))
    ground_truth = sitk.GetArrayFromImage(ground_truth)
    ground_truth = np.transpose(ground_truth, [1, 0, 2])  # rows, cols, z
    ground_truth = np.stack([cv2.resize(ground_truth[i], (512, 512)) for i in range(512)], axis=0)
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
        for pat_id in ['59']:
            for processor in models[key]['image_processors']:
                processor.get_niftii_info(image_handle)
                print('Performing pre process {}'.format(processor))
                images, ground_truth = processor.pre_process(images, ground_truth)
            if key == 'liver':
                pred = model.predict(images)
            else:
                pred = predict(images, model)

            for processor in models[key]['image_processors'][::-1]:  # In reverse now
                print('Performing post process {}'.format(processor))
                images, pred, ground_truth = processor.post_process(images, pred, ground_truth)
            for processor in models[key]['prediction_processors']:
                processor.get_niftii_info(image_handle)
                print('Performing prediction process {}'.format(processor))
                images, pred, ground_truth = processor.post_process(images, pred, ground_truth)
            pred = pred[..., -1]
            pred = np.stack([cv2.resize(pred[i], (512, 79)) for i in range(512)], axis=0)
            pred = np.transpose(pred, [1, 0, 2])  # rows, cols, z
            if key != 'liver':
                pred[pred > 0] = 2
            pred_handle = sitk.GetImageFromArray(pred)
            pred_handle.SetSpacing(image_handle.GetSpacing())
            pred_handle.SetOrigin(image_handle.GetOrigin())
            pred_handle.SetDirection(image_handle.GetDirection())
            if key == 'liver':
                sitk.WriteImage(sitk.Cast(pred_handle, sitk.sitkUInt8), os.path.join(out_path, 'test-volume-59-liver.nii'))
            else:
                sitk.WriteImage(sitk.Cast(pred_handle, sitk.sitkUInt8), os.path.join(out_path, 'test-segmentation-59.nii'))
            break