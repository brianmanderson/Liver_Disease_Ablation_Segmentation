__author__ = 'Brian M Anderson'
# Created on 1/17/2020

import os, sys
from Base_Deeplearning_Code.Data_Generators.Generators import Data_Generator_Class
from Base_Deeplearning_Code.Data_Generators.Image_Processors import *


def return_generators(get_mean_std=False, get_size=False, inverse_images=False, liver_norm=False):
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
    mean_val = 67
    std_val = 36
    image_num = 1
    expansion = 5
    lower_bound = -7
    upper_bound = 7
    if get_mean_std:
        mean_val = 0
        std_val = 1
        lower_bound = -np.inf
        upper_bound = np.inf
    if liver_norm:
        normalize = Normalize_to_Liver(0.2, upper=False)
    else:
        normalize = Normalize_Images(mean_val=mean_val,std_val=std_val)
    image_processors_train = [normalize,Ensure_Image_Proportions(512, 512),
                              Add_Noise_To_Images(by_patient=True, variation=np.arange(start=0, stop=0.3, step=0.01)),
                              Threshold_Images(lower_bound=lower_bound, upper_bound=upper_bound,
                                               inverse_image=inverse_images),
                              Annotations_To_Categorical(num_of_classes=num_classes)
                              ]
    image_processors_test = [normalize, Ensure_Image_Proportions(512, 512),
                             Threshold_Images(lower_bound=lower_bound, upper_bound=upper_bound,
                                              inverse_image=inverse_images),
                             Annotations_To_Categorical(num_of_classes=num_classes)
                             ]
    train_generator = Data_Generator_Class(by_patient=True,num_patients=image_num, whole_patient=True, shuffle=True,
                                           data_paths=paths, expansion=expansion,
                                           image_processors=image_processors_train)
    validation_generator = Data_Generator_Class(by_patient=True,num_patients=image_num, whole_patient=True, shuffle=False,
                                                data_paths=paths_validation_generator, expansion=expansion,
                                                image_processors=image_processors_test)
    # x,y = train_generator.__getitem__(0)
    if get_mean_std:
        livers = []
        diseases = []
        for i in range(len(train_generator)):
            print(i)
            # print(train_generator.generator.image_list)
            x, y = train_generator.__getitem__(i)
            data = x[y[..., 2] == 1][..., 0]
            livers.append(np.mean(x[y[...,1]==1]))
            diseases.append(np.mean(x[y[...,2]==1]))
            # print(np.mean(data))
            if i == 0:
                output = data
            else:
                output = np.append(output, data, axis=0)
        print(np.mean(output, axis=0))
        print(np.std(output, axis=0))
        print(np.median(output, axis=0))
        fid = open(os.path.join('.','output.txt'),'w+')
        fid.write('Liver,disease\n')
        for i in range(len(livers)):
            fid.write('{},{}\n'.format(livers[i],diseases[i]))
        fid.close()
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
    # return_generators(False, liver_norm=True)
    pass
