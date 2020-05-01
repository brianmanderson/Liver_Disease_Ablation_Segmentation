__author__ = 'Brian M Anderson'
# Created on 1/17/2020

from Base_Deeplearning_Code.Data_Generators.TFRecord_to_Dataset_Generator import Data_Generator_Class
from Base_Deeplearning_Code.Data_Generators.Image_Processors_Module.Image_Processors_DataSet import *
from Base_Deeplearning_Code.Models.TF_Keras_Models import my_UNet, Return_Layer_Functions, return_hollow_layers_dict
from Return_Morfeus_Base_Paths import return_paths, os
from _collections import OrderedDict


def get_layers_dict(layers=1, filters=16, max_filters=np.inf, **kwargs):
    lc = Return_Layer_Functions(kernel=(3,3,3),strides=(1,1,1),padding='same',batch_norm=True,
                                pooling_type='Max', pool_size=(2,2,2))
    dfkw = {'padding':'same','batch_norm':True, 'activation':'elu'}
    num_conv_blocks = 2
    layers_dict = return_hollow_layers_dict(layers)
    pool = (4,4,4)
    for layer in range(layers - 1):
        encoding = []
        for _ in range(num_conv_blocks):
            encoding.append(lc.atrous_layer(filters, **dfkw))
        if layer != 0:
            encoding = [lc.residual_layer(encoding, **dfkw)]
        layers_dict['Layer_' + str(layer)]['Encoding'] = encoding
        if filters < max_filters:
            filters = int(filters*2)
        decoding = []
        for _ in range(num_conv_blocks):
            decoding.append(lc.atrous_layer(filters, **dfkw))
        decoding = [lc.residual_layer(decoding, **dfkw)]
        layers_dict['Layer_' + str(layer)]['Decoding'] = decoding
        layers_dict['Layer_' + str(layer)]['Pooling']['Encoding'] = lc.pooling_layer(pool_size=pool)
        layers_dict['Layer_' + str(layer)]['Pooling']['Decoding'] = lc.upsampling_layer(pool_size=pool)
        pool = (2,2,2)
    block = []
    for _ in range(num_conv_blocks):
        block.append([lc.atrous_layer(filters, **dfkw)])
    block = [lc.residual_layer(block, **dfkw)]
    layers_dict['Base'] = block
    layers_dict['Final_Steps'] = [lc.convolution_layer(16, **dfkw),
                                  lc.convolution_layer(2, batch_norm=False, activation='softmax')]
    return layers_dict


def return_base_dict(step_size_factor=10, step_size_add=3, save_a_model=False,optimizer='Adam'):
    base_dict = lambda min_lr, max_lr, layers, filters, max_filters: \
        OrderedDict({'Architecture':{'model_name':'','layers': layers,
                                     'filters':filters, 'max_filters':max_filters},
                     'Hyper_Parameters':{'Save_Model':save_a_model,'Optimizer':optimizer, 'min_lr':min_lr,
                                         'max_lr':max_lr, 'step_size_factor': step_size_factor,
                                         'step_size_add':step_size_add,
                                         }
                     })
    return base_dict


def return_generators(batch_size=16, wanted_keys={'inputs':['image','mask','sum_vals'],'outputs':['annotation']}, return_test=False):
    base_path, morfeus_drive = return_paths()
    if not os.path.exists(base_path):
        print('{} does not exist'.format(base_path))
    train_path = [os.path.join(base_path, 'Train', 'Train.tfrecord')]
    validation_path = [os.path.join(base_path, 'Validation', 'Validation.tfrecord')]
    test_path = [os.path.join(base_path, 'Test', 'Test.tfrecord')]

    train_generator = Data_Generator_Class(record_names=train_path)
    validation_generator = Data_Generator_Class(record_names=validation_path)
    num_classes = 2
    train_processors = validation_processors = [
        Expand_Dimensions(axis=-1, on_images=True, on_annotations=True),
        Ensure_Image_Proportions(image_rows=120, image_cols=120),
        Return_Add_Mult_Disease(),
        Cast_Data({'image': 'float32', 'annotation': 'float32'}),
                        ]
    train_processors += [
        {'shuffle':len(train_generator)//10},
        {'batch':batch_size},
        Return_Outputs(wanted_keys),
        {'repeat'}
    ]
    validation_processors += [
        Return_Outputs(wanted_keys),
        {'repeat'}
    ]
    train_generator.compile_data_set(image_processors=train_processors, debug=False)
    validation_generator.compile_data_set(image_processors=validation_processors)
    data_set = iter(validation_generator.data_set)
    data = next(data_set)
    return base_path, morfeus_drive, train_generator, validation_generator


if __name__ == '__main__':
    # return_generators()
    pass
