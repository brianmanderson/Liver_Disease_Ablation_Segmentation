__author__ = 'Brian M Anderson'
# Created on 1/17/2020

import os, sys
from Base_Deeplearning_Code.Data_Generators.Generators import Train_Data_Generator3D, Image_Clipping_and_Padding
from Base_Deeplearning_Code.Data_Generators.Image_Processors import *
from Base_Deeplearning_Code.Keras_Utils.Keras_Utilities import dice_coef_3D_np, ModelCheckpoint_new, get_available_gpus, save_obj,load_obj, \
    remove_non_liver, weighted_categorical_crossentropy, weighted_categorical_crossentropy_masked, dice_coef_3D, np, EarlyStopping_BMA


try:
    base = r'\\mymdafiles\di_data1'
    base_path = os.path.join(base,r'Morfeus\BMAnderson\CNN\Data\Data_Liver\Liver_Disease_Ablation_Segmentation\Niftii_Data')
    os.listdir(base_path)
except:
    base = os.path.join('..', '..', '..', '..', '..', '..', '..')
    base_path = os.path.join(base, 'Liver_Segments')
morfeus_drive = os.path.abspath(os.path.join(base,'Morfeus','BMAnderson','CNN','Data','Data_Liver','Liver_Segments'))
paths = [os.path.join(base_path, 'Train', 'Single_Images3D')]
paths_test_generator = [os.path.join(base_path, 'Test', 'Single_Images3D')]
paths_validation_generator = [os.path.join(base_path, 'Validation', 'Single_Images3D')]

num_classes = 9
batch_size = 1
mean_val = 0
std_val = 1
image_num = 1
desired_output = None
expansion = 5
clip = [0, 0, 0]

image_processors_train = [Normalize_Images(mean_val=mean_val, std_val=std_val), Ensure_Image_Proportions(512, 512),
                          Add_Noise_To_Images(by_patient=True, variation=np.arange(start=0, stop=0.3, step=0.01)),
                          Threshold_Images(lower_bound=-3.55, upper_bound=3.55),
                          Annotations_To_Categorical(num_of_classes=3)]  #
image_processors_train = [Normalize_Images(mean_val=mean_val, std_val=std_val), Ensure_Image_Proportions(512, 512),
                          Annotations_To_Categorical(num_of_classes=3)]  #
image_processors_test = [Normalize_Images(mean_val=mean_val, std_val=std_val), Ensure_Image_Proportions(512, 512),
                         Threshold_Images(lower_bound=-3.55, upper_bound=3.55),
                         Annotations_To_Categorical(num_of_classes=3)]
train_generator = Train_Data_Generator3D(batch_size=image_num,
                                         whole_patient=True, shuffle=False, num_patients=batch_size,
                                         data_paths=paths, clip=clip,
                                         flatten=False, expansion=expansion, three_layer=False,
                                         is_test_set=True, all_for_one=False, verbose=False, output_size=desired_output,
                                         write_predictions=False, image_processors=image_processors_train)
x,y = train_generator.__getitem__(0)
xxx = 1