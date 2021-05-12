__author__ = 'Brian M Anderson'
# Created on 5/11/2021
from Deep_Learning.Base_Deeplearning_Code.Data_Generators.TFRecord_to_Dataset_Generator import DataGeneratorClass, plot_scroll_Image
import Deep_Learning.Base_Deeplearning_Code.Data_Generators.Image_Processors_Module.src.Processors.TFDataSetProcessors as Processors
from Deep_Learning.Utils.Return_Paths import return_paths
import os


def return_generators(is_2D=True, cache=False, batch=32):
    if is_2D:
        records_add = '2D_'
    else:
        records_add = '3D_'
    base_path, morfeus_drive, _ = return_paths()
    train_path = os.path.join(base_path, 'Records_1mm', '{}Train_32_Records'.format(records_add))
    validation_path = os.path.join(base_path, 'Records_1mm', 'Validation_Records')
    train_generator = DataGeneratorClass(record_paths=[train_path])
    train_processors = [
        Processors.ExpandDimension(image_keys=('image',), axis=-1),
        Processors.Threshold_Images(keys=('image',), lower_bounds=(-10,), upper_bounds=(10,), divides=(False,)),
        Processors.MultiplyImagesByConstant(keys=('image',), values=(1/20,)),
        Processors.AddConstantToImages(keys=('image',), values=(0.5,)),
        Processors.CreateNewKeyFromArgSum(guiding_keys=('annotation', ), new_keys=('mask', )),
        Processors.MaskOneBasedOnOther(guiding_keys=('mask', ), changing_keys=('image', ),
                                       guiding_values=(0, ), mask_values=(0, ), methods=('equal_to', )),
        Processors.ReturnSingleChannel(image_keys=('annotation', ), channels=(1, ), out_keys=('annotation', )),
        Processors.ToCategorical(annotation_keys=('annotation', ), number_of_classes=(2, )),
        Processors.Cast_Data(keys=('annotation', 'image', 'mask'), dtypes=('float16', 'float16', 'int32'))
    ]
    if cache:
        train_processors += [{'cache': train_path}]
    train_processors += [
        Processors.Flip_Images(keys=['image', 'mask', 'annotation'], flip_lr=True, flip_up_down=True,
                               flip_3D_together=True, flip_z=True),
        Processors.ReturnOutputs(input_keys=('image', 'mask'), output_keys=('annotation',)),
    ]
    if is_2D:
        train_processors += [
            {'shuffle': len(train_generator)},
            {'batch': batch}, {'batch': 1},
            {'repeat'}
        ]
    else:
        train_processors += [
            {'shuffle': len(train_generator)},
            {'batch': 1},
            {'repeat'}
        ]
    train_generator.compile_data_set(image_processors=train_processors, debug=False)

    validation_generator = DataGeneratorClass(record_paths=[validation_path])
    validation_processors = [
        Processors.ExpandDimension(image_keys=('image',), axis=-1),
        Processors.Threshold_Images(keys=('image',), lower_bounds=(-10,), upper_bounds=(10,), divides=(False,)),
        Processors.MultiplyImagesByConstant(keys=('image',), values=(1/20,)),
        Processors.AddConstantToImages(keys=('image',), values=(0.5,)),
        Processors.CreateNewKeyFromArgSum(guiding_keys=('annotation', ), new_keys=('mask', )),
        Processors.MaskOneBasedOnOther(guiding_keys=('mask', ), changing_keys=('image', ),
                                       guiding_values=(0, ), mask_values=(0, ), methods=('equal_to', )),
        Processors.ReturnSingleChannel(image_keys=('annotation', ), channels=(1, ), out_keys=('annotation', )),
        Processors.ToCategorical(annotation_keys=('annotation', ), number_of_classes=(2, )),
        Processors.Cast_Data(keys=('annotation', 'image', 'mask'), dtypes=('float16', 'float16', 'int32')),
        Processors.ReturnOutputs(input_keys=('image', 'mask'), output_keys=('annotation',))
    ]
    if cache:
        validation_processors += [{'cache': train_path}]
    validation_processors += [
        {'shuffle': len(train_generator)},
        {'batch': 1},
        {'repeat'}
    ]
    validation_generator.compile_data_set(image_processors=validation_processors, debug=False)
    return train_generator, validation_generator


if __name__ == '__main__':
    pass
    # return_generators()
