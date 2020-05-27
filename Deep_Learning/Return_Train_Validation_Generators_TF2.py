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
        val = ''
        if feature in run_data:
            val = run_data[feature]
            if type(val) is bool:
                val = int(val)
        out_dict[feature] = [val]
    out_features = [i for i in out_dict.keys()]
    return pd.DataFrame(out_dict), out_features


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
    return df


def return_hyper_parameters():
    HP_NUM_LAYERS = hp.HParam('layers', hp.Discrete([2, 3, 4]))
    HP_FILTERS = hp.HParam('filters', hp.Discrete([8, 16]))
    HP_MAX_FILTERS = hp.HParam('max_filters', hp.Discrete([32, 64, 128]))
    hp_dict = {'layers':HP_NUM_LAYERS, 'filters':HP_FILTERS, 'max_filters':HP_MAX_FILTERS}
    return hp_dict


def return_hparams(run_data, features_list, excluded_keys=['iteration','save']):
    hparams = None
    for layer_key in features_list:
        break_out = False
        for exclude in excluded_keys:
            if layer_key.lower().find(exclude) != -1:
                break_out = True
        if break_out:
            continue
        if layer_key in run_data.keys():
            if hparams is None:
                hparams = OrderedDict()
            hparams[hp.HParam(layer_key, hp.Discrete([run_data[layer_key]]))] = run_data[layer_key]
    return hparams


def return_best_dictionary(base_dict):
    dictionary = [
        base_dict(max_conv_blocks=4, layers=2, num_conv_blocks=3, conv_lambda=1, filters=32, max_filters=128,
                  min_lr=4e-7, max_lr=1.75e-3),
        # base_dict(max_conv_blocks=4, layers=3, num_conv_blocks=3, conv_lambda=1, filters=32, max_filters=128,
        #           min_lr=4e-7, max_lr=1.75e-3),
                  ]
    return dictionary


def return_dictionary(base_dict):
    dictionary = [
        base_dict(max_conv_blocks=4, layers=2, num_conv_blocks=3, conv_lambda=1, filters=32, max_filters=128,
                  min_lr=4e-7, max_lr=1.75e-3),
        base_dict(max_conv_blocks=4, layers=3, num_conv_blocks=3, conv_lambda=1, filters=32, max_filters=128,
                  min_lr=4e-7, max_lr=1.75e-3),
        base_dict(max_conv_blocks=4, layers=4, num_conv_blocks=3, conv_lambda=1, filters=32, max_filters=128,
                  min_lr=4e-7, max_lr=1.2e-3),
        base_dict(max_conv_blocks=4, layers=4, num_conv_blocks=3, conv_lambda=2, filters=32, max_filters=128,
                  min_lr=4e-7, max_lr=3e-4),

        base_dict(max_conv_blocks=6, layers=2, num_conv_blocks=3, conv_lambda=2, filters=32, max_filters=128,
                  min_lr=4e-7, max_lr=1.6e-3),
        base_dict(max_conv_blocks=6, layers=3, num_conv_blocks=3, conv_lambda=1, filters=32, max_filters=128,
                  min_lr=4e-7, max_lr=1.6e-3),
        base_dict(max_conv_blocks=6, layers=3, num_conv_blocks=3, conv_lambda=2, filters=32, max_filters=128,
                  min_lr=4e-7, max_lr=1.6e-3),
        base_dict(max_conv_blocks=6, layers=4, num_conv_blocks=3, conv_lambda=1, filters=32, max_filters=128,
                  min_lr=4e-7, max_lr=1.2e-3),
        base_dict(max_conv_blocks=6, layers=4, num_conv_blocks=3, conv_lambda=2, filters=32, max_filters=128,
                  min_lr=4e-7, max_lr=3e-4),

        base_dict(max_conv_blocks=8, layers=3, num_conv_blocks=3, conv_lambda=2, filters=32, max_filters=128,
                  min_lr=4e-7, max_lr=3e-4)
                  ]
    return dictionary


def get_atrous_layers_dict(layers=1, filters=16, max_filters=np.inf, num_conv_blocks=2, conv_lambda=0, bn_before_activation=True, **kwargs):
    lc = Return_Layer_Functions(kernel=(3,3,3),strides=(1,1,1),padding='same',batch_norm=True,
                                pooling_type='Max', pool_size=(2,2,2), bn_before_activation=bn_before_activation)
    dfkw = {'padding':'same','batch_norm':True, 'activation':'elu'}
    layers_dict = {}
    for layer in range(layers):
        encoding = []
        for _ in range(num_conv_blocks):
            encoding.append(lc.atrous_layer(filters, **dfkw))
        if layer != 0:
            encoding = [lc.residual_layer(encoding, **dfkw)]
        layers_dict['Layer_{}'.format(layer)] = {'Encoding':encoding}
        if filters < max_filters:
            filters = int(filters*2)
        num_conv_blocks += conv_lambda
    final_steps = [lc.convolution_layer(32, **dfkw),
                   lc.convolution_layer(2, batch_norm=False, activation='softmax')]
    layers_dict['Final_Steps'] = final_steps
    return layers_dict


def get_layers_dict(layers=1, filters=16, max_filters=np.inf, conv_lambda=0, num_conv_blocks=2, max_conv_blocks=4, atrous=True,**kwargs):
    lc = Return_Layer_Functions(kernel=(3,3,3),strides=(1,1,1),padding='same',batch_norm=True,
                                pooling_type='Max', pool_size=(2,2,2), bn_before_activation=True)
    if atrous:
        factor = 1
        key = 'atrous'
        block = lc.atrous_layer
    else:
        key = 'convolution'
        factor = 2
        block = lc.convolution_layer
    dfkw = {'padding':'same','batch_norm':True, 'activation':'elu'}
    layers_dict = return_hollow_layers_dict(layers)
    pool = (2, 2, 2)
    final_steps = [lc.convolution_layer(filters, **dfkw),
                   lc.convolution_layer(2, batch_norm=False, activation='softmax')]
    layers_dict['Final_Steps'] = final_steps
    first = True
    for layer in range(layers - 1):
        layers_dict['Layer_' + str(layer)]['Encoding'] = []
        encoding = []
        subtract = 1 if num_conv_blocks % factor != 0 else 0
        for i in range((num_conv_blocks // factor - subtract) * factor):
            encoding.append(block(filters, **dfkw))
            if (i + 1) % factor == 0:
                if not first:
                    if atrous:
                        encoding[-1][key]['activation'][-1] = None
                    else:
                        encoding[-1][key]['activation'] = None
                    encoding = [lc.residual_layer(encoding, **dfkw)]
                layers_dict['Layer_' + str(layer)]['Encoding'] += encoding
                encoding = []
                first = False
        if num_conv_blocks % factor != 0:
            encoding = []
            for i in range(num_conv_blocks % factor + factor):
                encoding.append(block(filters, **dfkw))
            if not first:
                if atrous:
                    encoding[-1][key]['activation'][-1] = None
                else:
                    encoding[-1][key]['activation'] = None
                encoding = [lc.residual_layer(encoding, **dfkw)]
            layers_dict['Layer_' + str(layer)]['Encoding'] += encoding
        first = False
        layers_dict['Layer_' + str(layer)]['Pooling']['Decoding'] = [lc.upsampling_layer(pool_size=pool),
                                                                     lc.convolution_layer(filters, **dfkw)]
        if filters < max_filters:
            filters = int(filters*2)
        layers_dict['Layer_' + str(layer)]['Pooling']['Encoding'] = lc.convolution_layer(filters, strides=(2,2,2), **dfkw)
        layers_dict['Layer_' + str(layer)]['Decoding'] = []
        decoding = []
        for i in range((num_conv_blocks // factor - subtract) * factor):
            decoding.append(block(filters, **dfkw))
            if (i + 1) % factor == 0:
                if not first:
                    if atrous:
                        decoding[-1][key]['activation'][-1] = None
                    else:
                        decoding[-1][key]['activation'] = None
                    decoding = [lc.residual_layer(decoding, **dfkw)]
                layers_dict['Layer_' + str(layer)]['Decoding'] += decoding
                decoding = []
                first = False
        if num_conv_blocks % factor != 0:
            decoding = []
            for i in range(num_conv_blocks % factor + factor):
                decoding.append(block(filters, **dfkw))
            if atrous:
                decoding[-1][key]['activation'][-1] = None
            else:
                decoding[-1][key]['activation'] = None
            decoding = [lc.residual_layer(decoding, **dfkw)]
            layers_dict['Layer_' + str(layer)]['Decoding'] += decoding
        num_conv_blocks += conv_lambda
        num_conv_blocks = min([num_conv_blocks, max_conv_blocks])
    base = []
    subtract = 1 if num_conv_blocks % factor != 0 else 0
    layers_dict['Base'] = []
    for i in range((num_conv_blocks // factor - subtract) * factor):
        base.append(block(filters, **dfkw))
        if (i + 1) % factor == 0:
            if not first:
                if atrous:
                    base[-1][key]['activation'][-1] = None
                else:
                    base[-1][key]['activation'] = None
                base = [lc.residual_layer(base, **dfkw)]
            layers_dict['Base'] += base
            base = []
            first = False
    if num_conv_blocks % factor != 0:
        base = []
        for i in range(num_conv_blocks % factor + factor):
            base.append(block(filters, **dfkw))
        if not first:
            if atrous:
                base[-1][key]['activation'][-1] = None
            else:
                base[-1][key]['activation'] = None
            base = [lc.residual_layer(base, **dfkw)]
        layers_dict['Base'] += base
    return layers_dict


def get_layers_dict_new(layers=1, filters=16, max_filters=np.inf, conv_lambda=0, num_conv_blocks=2, max_conv_blocks=4, num_classes=2,**kwargs):
    lc = Return_Layer_Functions(kernel=(3,3,3),strides=(1,1,1),padding='same',batch_norm=True,
                                pooling_type='Max', pool_size=(2,2,2), bn_before_activation=False)
    block = lc.convolution_layer
    dfkw = {'padding':'same','batch_norm':True, 'activation':'elu'}
    layers_dict = return_hollow_layers_dict(layers)
    pool = (2, 2, 2)
    final_steps = [lc.convolution_layer(filters, **dfkw),
                   lc.convolution_layer(num_classes, batch_norm=False, activation='softmax')]
    layers_dict['Final_Steps'] = final_steps
    first = True
    for layer in range(layers - 1):
        layers_dict['Layer_' + str(layer)]['Encoding'] = []
        encoding = []
        for i in range(num_conv_blocks):
            encoding.append(block(filters, **dfkw))
        if first:
            layers_dict['Layer_' + str(layer)]['Encoding'] = [lc.batch_norm_layer(), encoding[0]]
            del encoding[0]
        if encoding:
            split = np.array_split(encoding,len(encoding)//2)
            for array in split:
                array[-1] = block(filters, activation=None, batch_norm=False)
                array = [lc.residual_layer(array, **dfkw), lc.batch_norm_layer()]
                layers_dict['Layer_' + str(layer)]['Encoding'] += array
        first = False
        layers_dict['Layer_' + str(layer)]['Pooling']['Decoding'] = [lc.upsampling_layer(pool_size=pool),
                                                                     lc.convolution_layer(filters, **dfkw)]
        if filters < max_filters:
            filters = int(filters*2)
        layers_dict['Layer_' + str(layer)]['Pooling']['Encoding'] = lc.convolution_layer(filters, strides=(2,2,2), **dfkw)
        layers_dict['Layer_' + str(layer)]['Decoding'] = []
        decoding = []
        for i in range(num_conv_blocks):
            decoding.append(block(filters, **dfkw))
        if decoding:
            split = np.array_split(decoding,len(decoding)//2)
            for array in split:
                array[-1] = block(filters, activation=None, batch_norm=False)
                array = [lc.residual_layer(array, **dfkw), lc.batch_norm_layer()]
                layers_dict['Layer_' + str(layer)]['Decoding'] += array
        num_conv_blocks += conv_lambda
        num_conv_blocks = min([num_conv_blocks, max_conv_blocks])
    base = []
    for i in range(num_conv_blocks):
        base.append(block(filters, **dfkw))
    if first:
        layers_dict['Base'] = [lc.batch_norm_layer(), base[0]]
        del base[0]
    if base:
        split = np.array_split(base, len(base) // 2)
        for array in split:
            array[-1] = block(filters, activation=None, batch_norm=False)
            array = [lc.residual_layer(array, **dfkw), lc.batch_norm_layer()]
            layers_dict['Base'] += array
    return layers_dict


def return_base_dict(step_size_factor=10, save_a_model=False,optimizer='Adam'):
    base_dict = lambda min_lr, max_lr, layers, num_conv_blocks, max_conv_blocks, conv_lambda, filters, max_filters: \
        OrderedDict({'atrous':False, 'layers': layers,'num_conv_blocks':num_conv_blocks, 'max_conv_blocks':max_conv_blocks,
                     'conv_lambda':conv_lambda, 'filters':filters, 'max_filters':max_filters,
                     'Save_Model':save_a_model,'Optimizer':optimizer, 'min_lr':min_lr,
                     'max_lr':max_lr, 'step_size_factor': step_size_factor
                     })
    return base_dict


def return_generators(batch_size=16, wanted_keys={'inputs':['image','mask'],'outputs':['annotation']},
                      add='', is_test=False, cache=True, validation_name='Validation',cache_add=''):
    base_path, morfeus_drive = return_paths()
    if not os.path.exists(base_path):
        print('{} does not exist'.format(base_path))
    train_path = [os.path.join(base_path, 'Train', 'Train{}.tfrecord'.format(add))]
    validation_path = [os.path.join(base_path, 'Validation', '{}.tfrecord'.format(validation_name))]
    ext = 'Validation'
    if is_test:
        validation_path = [os.path.join(base_path, 'Test', 'Test.tfrecord')]
        ext = 'Test'

    train_generator = Data_Generator_Class(record_names=train_path)
    validation_generator = Data_Generator_Class(record_names=validation_path, in_parallel=True)
    train_processors, validation_processors = [], []
    base_processors = [
        Expand_Dimensions(axis=-1, on_images=True, on_annotations=True),
                        ]
    train_processors += base_processors
    validation_processors += base_processors
    train_processors += [
        Ensure_Image_Proportions(image_rows=120, image_cols=120),
        Return_Add_Mult_Disease(change_background=True),
        Cast_Data({'image': 'float16', 'annotation': 'float16', 'mask': 'int32'}),
        # Threshold_Images(lower_bound=-10, upper_bound=10),
        {'cache': os.path.join(base_path,'Train{}'.format(add))},
        Flip_Images(keys=['image','mask','annotation'], flip_lr=True, flip_up_down=True, flip_3D_together=True, flip_z=True),
        Return_Outputs(wanted_keys),
        {'shuffle': len(train_generator)//3},
        {'batch': batch_size},
        {'repeat'}
    ]
    validation_processors += [
        Return_Add_Mult_Disease(change_background=True),
        # Threshold_Images(lower_bound=-10, upper_bound=10),
        Cast_Data({'image': 'float16', 'annotation': 'float16', 'mask': 'int32'}),
        Return_Outputs(wanted_keys),
        {'batch':1}]
    if cache:
        validation_processors += [
        {'cache': os.path.join(base_path,'{}{}{}'.format(ext,add,cache_add))}
        ]
    validation_processors += [
        {'repeat'}
    ]
    train_generator.compile_data_set(image_processors=train_processors, debug=False)
    validation_generator.compile_data_set(image_processors=validation_processors)
    for generator in [train_generator, validation_generator]: #
        data_set = iter(generator.data_set)
        for _ in range(len(generator)):
            x, y = next(data_set)
            print(x[0].shape)
    #     print(data[1][0].shape)
    # data = next(data_set)
    return base_path, morfeus_drive, train_generator, validation_generator


if __name__ == '__main__':
    # return_generators(add='_32')
    pass
