__author__ = 'Brian M Anderson'
# Created on 1/17/2020

import os, sys
from Base_Deeplearning_Code.Data_Generators.Generators import Train_Data_Generator3D
from Base_Deeplearning_Code.Data_Generators.Image_Processors import *


def return_generators(get_mean_std=False, get_size=False):
    try:
        base = r'\\mymdafiles\di_data1'
        base_path = os.path.join(base,r'Morfeus\BMAnderson\CNN\Data\Data_Liver\Liver_Disease_Ablation_Segmentation\Niftii_Data')
        os.listdir(base_path)
    except:
        base = os.path.join('..', '..', '..', '..', '..', '..', '..')
        base_path = os.path.join(base, 'Liver_GTV_Ablation')
    morfeus_drive = os.path.abspath(os.path.join(base,'Morfeus','BMAnderson','CNN','Data','Data_Liver','Liver_Disease_Ablation_Segmentation'))
    paths = [os.path.join(base_path, 'Train', 'Single_Images3D')]
    paths_validation_generator = [os.path.join(base_path, 'Validation', 'Single_Images3D')]

    num_classes = 3
    batch_size = 1
    mean_val = 67.47
    std_val = 36.72
    image_num = 1
    expansion = 5
    lower_bound = -3.55
    upper_bound = 3.55
    if get_mean_std:
        mean_val = 0
        std_val = 1
        lower_bound = -np.inf
        upper_bound = np.inf

    image_processors_train = [Normalize_Images(mean_val=mean_val, std_val=std_val), Ensure_Image_Proportions(512, 512),
                              Add_Noise_To_Images(by_patient=True, variation=np.arange(start=0, stop=0.3, step=0.01)),
                              Threshold_Images(lower_bound=lower_bound, upper_bound=upper_bound),
                              Annotations_To_Categorical(num_of_classes=num_classes)]
    image_processors_test = [Normalize_Images(mean_val=mean_val, std_val=std_val), Ensure_Image_Proportions(512, 512),
                             Threshold_Images(lower_bound=lower_bound, upper_bound=upper_bound),
                             Annotations_To_Categorical(num_of_classes=num_classes)]
    train_generator = Train_Data_Generator3D(batch_size=image_num,whole_patient=True, shuffle=False,
                                             num_patients=batch_size, data_paths=paths, expansion=expansion,
                                             three_layer=False, is_test_set=True,
                                             image_processors=image_processors_train)
    validation_generator = Train_Data_Generator3D(batch_size=image_num,whole_patient=True, shuffle=False,
                                                  num_patients=batch_size, data_paths=paths_validation_generator,
                                                  expansion=expansion, three_layer=False, is_test_set=True,
                                                  image_processors=image_processors_test)
    if get_mean_std:
        for i in range(len(train_generator)):
            print(i)
            # print(train_generator.generator.image_list)
            x, y = train_generator.__getitem__(i)
            data = x[y[..., 2] == 1][..., 0]
            print(np.mean(data))
            if i == 0:
                output = data
            else:
                output = np.append(output, data, axis=0)
        print(np.mean(output, axis=0))
        print(np.std(output, axis=0))
        print(np.median(output, axis=0))
    if get_size:
        background = 0
        thing = 0
        total = 0
        for i in range(len(train_generator)):
            print(i)
            # print(train_generator.generator.image_list)
            x, y = train_generator.__getitem__(i)
            total += np.prod(y[...,0].shape)
            indexes = np.where(y[...,-1]==1)
            thing += len(indexes[0])
            indexes = np.where(y[...,-1]!=1)
            background += len(indexes[0])
        print(thing/total)
        print(background/total)
    return base_path, morfeus_drive, train_generator, validation_generator


if __name__ == '__main__':
    pass
