__author__ = 'Brian M Anderson'
# Created on 1/17/2020

from Base_Deeplearning_Code.Data_Generators.TFRecord_to_Dataset_Generator import Data_Generator_Class
from Base_Deeplearning_Code.Data_Generators.Image_Processors_Module.Image_Processors_DataSet import *
from Base_Deeplearning_Code.Models.TF_Keras_Models import my_UNet, Return_Layer_Functions, return_hollow_layers_dict
from Return_Morfeus_Base_Paths import return_paths, os
from _collections import OrderedDict
import pandas as pd
from tensorboard.plugins.hparams import api as hp


def return_current_df(run_data, features_list=['layers', 'filters', 'max_filters', 'min_lr', 'max_lr']):
    out_dict = OrderedDict()
    for feature in features_list:
        val = None
        if feature in run_data:
            val = run_data[feature]
        out_dict[feature] = [val]
    return pd.DataFrame(out_dict)


def return_pandas_df(excel_path, features_list=['layers','filters','max_filters','min_lr','max_lr']):
    if not os.path.exists(excel_path):
        out_dict = OrderedDict()
        out_dict['Trial_ID'] = []
        for key in features_list:
            out_dict[key] = []
        df = pd.DataFrame(out_dict)
        df.to_excel(excel_path, index=0)
    else:
        df = pd.read_excel(excel_path)
    out_features = [i for i in df.keys() if i not in ['Trial_ID']]
    return df, out_features


def return_hyper_parameters():
    HP_NUM_LAYERS = hp.HParam('layers', hp.Discrete([2, 3, 4]))
    HP_FILTERS = hp.HParam('filters', hp.Discrete([8, 16]))
    HP_MAX_FILTERS = hp.HParam('max_filters', hp.Discrete([32, 64, 128]))
    hp_dict = {'layers':HP_NUM_LAYERS, 'filters':HP_FILTERS, 'max_filters':HP_MAX_FILTERS}
    return hp_dict


def return_hparams(run_data):
    hparams = None
    for layer_key in run_data:
        data = run_data[layer_key]
        if type(data) is int:
            if hparams is None:
                hparams = {}
            hparams[hp.HParam(layer_key, hp.Discrete([run_data[layer_key]]))] = run_data[layer_key]
    return hparams


def return_dictionary(base_dict):
    dictionary = [
        base_dict(min_lr=1e-4, max_lr=8e-2, layers=2, filters=8, max_filters=32),
        base_dict(min_lr=1e-4, max_lr=8e-2, layers=2, filters=8, max_filters=64),
        base_dict(min_lr=1e-4, max_lr=8e-2, layers=2, filters=8, max_filters=128),
        base_dict(min_lr=1e-4, max_lr=8e-2, layers=2, filters=16, max_filters=32),
        base_dict(min_lr=1e-4, max_lr=8e-2, layers=2, filters=16, max_filters=64),
        base_dict(min_lr=1e-4, max_lr=8e-2, layers=2, filters=16, max_filters=128),

        base_dict(min_lr=1e-4, max_lr=8e-2, layers=3, filters=8, max_filters=32),
        base_dict(min_lr=1e-4, max_lr=8e-2, layers=3, filters=8, max_filters=64),
        base_dict(min_lr=1e-4, max_lr=8e-2, layers=3, filters=8, max_filters=128),
        base_dict(min_lr=1e-4, max_lr=8e-2, layers=3, filters=16, max_filters=32),
        base_dict(min_lr=1e-4, max_lr=8e-2, layers=3, filters=16, max_filters=64),
        base_dict(min_lr=1e-4, max_lr=8e-2, layers=3, filters=16, max_filters=128),

        base_dict(min_lr=1e-4, max_lr=8e-2, layers=4, filters=8, max_filters=32),
        base_dict(min_lr=1e-4, max_lr=8e-2, layers=4, filters=8, max_filters=64),
        base_dict(min_lr=1e-4, max_lr=8e-2, layers=4, filters=8, max_filters=128),
        base_dict(min_lr=1e-4, max_lr=8e-2, layers=4, filters=16, max_filters=32),
        base_dict(min_lr=1e-4, max_lr=8e-2, layers=4, filters=16, max_filters=64),
        base_dict(min_lr=1e-4, max_lr=8e-2, layers=4, filters=16, max_filters=128),
    ]
    return dictionary


def get_layers_dict(layers=1, filters=16, max_filters=np.inf, bn_before_activation=True, **kwargs):
    lc = Return_Layer_Functions(kernel=(3,3,3),strides=(1,1,1),padding='same',batch_norm=True,
                                pooling_type='Max', pool_size=(2,2,2), bn_before_activation=bn_before_activation)
    dfkw = {'padding':'same','batch_norm':True, 'activation':'elu'}
    num_conv_blocks = 2
    layers_dict = return_hollow_layers_dict(layers)
    pool = (2, 2, 2)
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


def return_base_dict(step_size_factor=10, save_a_model=False,optimizer='Adam'):
    base_dict = lambda min_lr, max_lr, layers, filters, max_filters: \
        OrderedDict({'model_name':'','layers': layers, 'filters':filters, 'max_filters':max_filters,
                     'Save_Model':save_a_model,'Optimizer':optimizer, 'min_lr':min_lr,
                     'max_lr':max_lr, 'step_size_factor': step_size_factor
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
    validation_generator = Data_Generator_Class(record_names=validation_path, in_parallel=True)
    num_classes = 2
    train_processors, validation_processors = [], []
    base_processors = [
        Expand_Dimensions(axis=-1, on_images=True, on_annotations=True),
        Cast_Data({'image': 'float32', 'annotation': 'float32'}),
                        ]
    train_processors += base_processors
    validation_processors += base_processors
    train_processors += [
        Ensure_Image_Proportions(image_rows=120, image_cols=120),
        Return_Add_Mult_Disease(),
        Cast_Data({'mask': 'float32', 'sum_vals': 'float32'}),
        {'shuffle':len(train_generator)//5},
        {'batch':batch_size},
        Return_Outputs(wanted_keys),
        {'repeat'},
        {'prefetch'}
    ]
    validation_processors += [
        Return_Add_Mult_Disease(),
        Cast_Data({'mask': 'float32', 'sum_vals': 'float32'}),
        {'batch':1},
        Return_Outputs(wanted_keys),
        {'repeat'}
    ]
    train_generator.compile_data_set(image_processors=train_processors, debug=False)
    validation_generator.compile_data_set(image_processors=validation_processors)
    # data_set = iter(validation_generator.data_set)
    # data = next(data_set)
    return base_path, morfeus_drive, train_generator, validation_generator


if __name__ == '__main__':
    # return_generators()
    pass
