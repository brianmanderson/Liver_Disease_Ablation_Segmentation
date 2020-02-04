import os, sys
from Base_Deeplearning_Code.Data_Generators.Generators import Image_Clipping_and_Padding
from Base_Deeplearning_Code.Data_Generators.Image_Processors import *
from keras.models import *
import tensorflow as tf
from Base_Deeplearning_Code.Keras_Utils.Keras_Utilities import dice_coef_3D_np, ModelCheckpoint_new, get_available_gpus, save_obj,load_obj, \
    remove_non_liver, weighted_categorical_crossentropy, weighted_categorical_crossentropy_masked, dice_coef_3D, np
from Base_Deeplearning_Code.Models.Keras_3D_Models import my_3D_UNet
from Base_Deeplearning_Code.Finding_Optimization_Parameters.LR_Finder import LearningRateFinder
from Return_Train_Validation_Generators import return_generators
from _collections import OrderedDict


def run_model(gpu=1,layers_dict=None, out_path='', train_generator=None, get_weights=False,
              mask_pred=False,batch_norm=False, mask_image=False,threshold_mask=3.55, weighted=False):
    G = get_available_gpus()
    if len(G) == 1:
        gpu = 0
    train_generator = Image_Clipping_and_Padding(layers_dict, train_generator, return_mask=mask_pred,
                                                 liver_box=True, mask_image=mask_image, threshold_value=threshold_mask,
                                                 remove_liver_layer=True)
    x,y = train_generator.__getitem__(0)
    if get_weights:
        print('Getting class weights')
        get_class_weights(train_generator)
        return None
    print(len(train_generator))
    with tf.device('/gpu:' + str(gpu)):
        gpu_options = tf.GPUOptions(allow_growth=True)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        K.set_session(sess)
        loss = 'categorical_crossentropy'
        if weighted:
            loss = weighted_categorical_crossentropy(np.asarray([1,500])) #categorical_crossentropy
        Model_class = my_3D_UNet(kernel=(3, 3, 3), layers_dict=layers_dict, pool_size=(2, 2, 2),custom_loss=loss,batch_norm=batch_norm,
                                 activation='elu', pool_type='Max',out_classes=2, mask_loss=False,mask_output=mask_pred)
        Model_val = Model_class.created_model

        LearningRateFinder(epochs=10, model=Model_val, metrics=['accuracy'],out_path=out_path,loss=loss,
                           train_generator=train_generator, lower_lr=1e-7, high_lr=1e-2)
        K.clear_session()


def get_layers_dict(layers=1, filters=16, conv_blocks=1, num_conv_blocks=None, num_atrous_blocks=4, max_blocks=2, max_filters=np.inf, **kwargs):
    atrous_rate = 2
    layers_dict = {}
    pool = (4, 4, 4)
    for layer in range(conv_blocks):
        conv_block = {'Channels': [filters]}
        if num_conv_blocks is not None:
            conv_blocks_total = [conv_block for _ in range(num_conv_blocks)]
        else:
            conv_blocks_total = [conv_block]
        layers_dict['Layer_' + str(layer)] = {'Encoding':conv_blocks_total,'Pooling':pool,'Decoding':conv_blocks_total}
        pool = (2, 2, 2)
        if filters < max_filters:
            filters = int(filters*2)
    pool = (2, 2, 2)
    for layer in range(conv_blocks,layers-1):
        atrous_block = {'Channels': [filters], 'Atrous_block': [atrous_rate], 'Kernel': [(3, 3, 3)]}
        layers_dict['Layer_' + str(layer)] = {'Encoding': [atrous_block for _ in range(num_atrous_blocks)], 'Pooling': pool,
                                              'Decoding': [atrous_block for _ in range(num_atrous_blocks)]}
        if filters < max_filters:
            filters = int(filters*2)
        num_atrous_blocks = min([(num_atrous_blocks) * 2,max_blocks])
    num_atrous_blocks = min([(num_atrous_blocks) * 2, max_blocks])
    atrous_block = {'Channels': [filters], 'Atrous_block': [atrous_rate],'Kernel': [(3, 3, 3)]}
    layers_dict['Base'] = {'Encoding':[atrous_block for _ in range(num_atrous_blocks)]}
    return layers_dict


def get_layers_dict_atrous(layers=1, filters=16, atrous_blocks=2, max_atrous_blocks=2, max_filters=np.inf,
                           atrous_rate=2, **kwargs):
    activation = 'relu'
    layers_dict = {}
    conv_block = lambda x: {'convolution': {'channels': x, 'kernel': (1, 1, 1), 'strides': (1, 1, 1),'activation':activation}}
    atrous_block = lambda x,y,z: {'atrous': {'channels': x, 'atrous_rate': y, 'activations': z}}
    previous_filters = [1]
    for layer in range(layers):
        encoding = [atrous_block(filters,atrous_rate,[activation for _ in range(atrous_rate)]) for _ in range(atrous_blocks)]
        if previous_filters[layer] != filters:
            encoding = [conv_block(filters)] + encoding
        previous_filters.append(filters)
        if filters < max_filters:
            filters = int(filters*2)
        layers_dict['Layer_' + str(layer)] = {'Encoding': encoding}
        if atrous_blocks < max_atrous_blocks:
            atrous_blocks = int(atrous_blocks*2)
    return layers_dict


def get_class_weights(train_generator, class_num=9, out_file_name=os.path.join('.','class_weights.npy')):
    data_dict = [0 for _ in range(class_num)]
    for i in range(len(train_generator)):
        print(i)
        x,y = train_generator.__getitem__(i)
        collapsed = np.argmax(y,axis=-1)
        collapsed = collapsed.flatten()
        for ii in range(len(data_dict)):
            values = np.where(collapsed==ii)
            if values:
                data_dict[ii] += len(values[0])
    print(data_dict)
    data_dict = np.asarray(data_dict)
    total = np.sum(data_dict)
    class_weights = [(total/i)/class_num for i in data_dict]
    print(class_weights)
    np.save(out_file_name,np.asarray(class_weights))
    return None


def return_dictionary(base_dict):
    dictionary = {
        1: [
            base_dict(2e-7, 7e-4, 16, 16)
        ],
        2: [
            base_dict(2e-7, 7e-4, 16, 16)
        ],
        3: [
            base_dict(2e-7, 7e-4, 16, 16)
        ],
        4: [
            base_dict(2e-7, 7e-4, 16, 16)
        ],
        5: [
            base_dict(2e-7, 7e-4, 16, 16)
        ],
        6: [
            base_dict(2e-7, 7e-4, 16, 16)
        ],
        7: [
            base_dict(2e-7, 7e-4, 16, 16)
        ],
        8: [
            base_dict(2e-7, 7e-4, 16, 16)
        ]
    }
    return dictionary


def return_things(run_data):
    things = []
    for top_key in ['Architecture']:
        model_info = run_data[top_key]
        for key in model_info:
            if model_info[key] is not 0 and model_info[key] is not False:
                if model_info[key] is True:
                    things.append('{}'.format(key))
                else:
                    things.append('{}_{}'.format(model_info[key],key))
    return things


def main():
    threshold_mask = -7
    inverse_images = False
    norm_to_liver = True
    if inverse_images:
        threshold_mask = 7
    _, _, train_generator, validation_generator = return_generators(inverse_images=inverse_images, liver_norm=norm_to_liver)
    x,y = train_generator.__getitem__(0)
    get_weights = False
    gpu = 2
    mask_image = False
    mask_pred = True
    batch_norm = False
    mask_loss = False
    base_dict = lambda min_lr, max_lr, filters, max_filters: \
        OrderedDict({'Architecture':{'layers': 0,'atrous_blocks': 1,'atrous_rate':1, 'max_atrous_blocks':1,
                                     'filters':filters, 'max_filters':max_filters,'layers_conv_blocks': 0,
                                     'conv_blocks': 0}
                     })
    for atrous_rate in [1, 2, 3, 4, 5]:
        overall_dictionary = return_dictionary(base_dict)
        for layer in overall_dictionary:
            data = overall_dictionary[layer]
            for run_data in data:
                run_data['Architecture']['atrous_rate'] = atrous_rate
                run_data['Architecture']['layers'] = layer
                layers_dict = get_layers_dict_atrous(**run_data['Architecture'])
                for iteration in [1,2,3]:
                    things = return_things(run_data)
                    out_path = os.path.join('..','..','Learning_Rates_Liver_Disease','All_Atrous')
                    for thing in things:
                        out_path = os.path.join(out_path,thing)
                    out_path = os.path.join(out_path,'{}_Iteration'.format(iteration))
                    if os.path.exists(out_path):
                        continue
                    print(out_path)
                    os.makedirs(out_path)
                    run_model(gpu=gpu, layers_dict=layers_dict, out_path=out_path, train_generator=train_generator,
                              get_weights=get_weights,batch_norm=batch_norm,
                              mask_image=mask_image, mask_pred=mask_pred, threshold_mask=threshold_mask)
                    if get_weights:
                        return None


if __name__ == '__main__':
    main()
