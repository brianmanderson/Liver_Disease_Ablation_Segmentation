__author__ = 'Brian M Anderson'
# Created on 1/18/2020
from Base_Deeplearning_Code.Data_Generators.Generators import Image_Clipping_and_Padding
from Base_Deeplearning_Code.Data_Generators.Return_Paths import *
from keras.models import *
from keras.layers.advanced_activations import PReLU
from keras.initializers import Constant
import tensorflow as tf
# from tensorflow.python.keras.losses import CategoricalCrossentropy
from Base_Deeplearning_Code.Keras_Utils.Keras_Utilities import balanced_cross_entropy, ModelCheckpoint_new, get_available_gpus, save_obj,load_obj, \
    remove_non_liver, weighted_categorical_crossentropy, categorical_crossentropy_masked, dice_coef_3D, np, EarlyStopping_BMA
from keras.callbacks import EarlyStopping
from Base_Deeplearning_Code.Callbacks.Visualizing_Model_Utils import TensorBoardImage
from Base_Deeplearning_Code.Plot_And_Scroll_Images.Plot_Scroll_Images import plot_scroll_Image
from Base_Deeplearning_Code.Data_Generators.Return_Paths import Path_Return_Class
from keras.optimizers import Adam
from Base_Deeplearning_Code.Models.Keras_3D_Models import my_3D_UNet
from Base_Deeplearning_Code.Callbacks.BMA_Callbacks import ModelCheckpoint_new
from Base_Deeplearning_Code.Cyclical_Learning_Rate.clr_callback import CyclicLR
from Return_Train_Validation_Generators import return_generators
from _collections import OrderedDict


def get_layers_dict(layers=1, filters=16, conv_blocks=1, num_atrous_blocks=4, max_blocks=2, max_filters=np.inf,
                    atrous_rate=1, max_atrous_rate=2, **kwargs):
    activation = {'activation':PReLU,'kwargs':{'alpha_initializer':Constant(0.25),'shared_axes':[1,2,3]}}
    activation = 'relu'
    layers_dict = {}
    conv_block = lambda x: {'convolution': {'channels': x, 'kernel': (3, 3, 3), 'strides': (1, 1, 1),'activation':activation}}
    strided_block = lambda x: {'convolution': {'channels': x, 'kernel': (3, 3, 3), 'strides': (2, 2, 2), 'activation':activation}}
    atrous_block = lambda x,y,z: {'atrous': {'channels': x, 'atrous_rate': y, 'activations': z}}
    for layer in range(conv_blocks,layers-1):
        encoding = [atrous_block(filters,atrous_rate,[activation for _ in range(atrous_rate)]) for _ in range(num_atrous_blocks)]
        atrous_block_dec = [atrous_block(filters,atrous_rate,[activation for _ in range(atrous_rate)]) for _ in range(num_atrous_blocks)]
        if layer == 0:
            encoding = [conv_block(filters)] + encoding
        if filters < max_filters:
            filters = int(filters*2)
        layers_dict['Layer_' + str(layer)] = {'Encoding': encoding,
                                              'Pooling':{'Encoding':[strided_block(filters)],'Pool_Size':(2,2,2)},
                                              'Decoding': atrous_block_dec}
        num_atrous_blocks = min([(num_atrous_blocks) * 2,max_blocks])
    num_atrous_blocks = min([(num_atrous_blocks) * 2, max_blocks])
    layers_dict['Base'] = {'Encoding':[atrous_block(filters,atrous_rate,[activation for _ in range(atrous_rate)]) for _ in range(num_atrous_blocks)]}
    return layers_dict


def get_layers_dict_atrous(layers=1, filters=16, atrous_blocks=2, max_atrous_blocks=2, max_filters=np.inf,
                           atrous_rate=2, **kwargs):
    activation = {'activation':PReLU,'kwargs':{'alpha_initializer':Constant(0.25),'shared_axes':[1,2,3]}}
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

def return_things(run_data):
    things = []
    for top_key in ['Architecture','Hyper_Parameters']:
        model_info = run_data[top_key]
        for key in model_info:
            if model_info[key] is not 0 and model_info[key] is not False:
                if model_info[key] is True:
                    things.append('{}'.format(key))
                else:
                    things.append('{}_{}'.format(model_info[key],key))
    return things


def return_dictionary_all(base_dict):
    dictionary = {
        3: [
            base_dict(1.5e-7, 4e-5, 32, 64, 1),
            base_dict(1.5e-7, 1e-4, 16, 32, 1),
            base_dict(2e-7, 4e-5, 32, 32, 1),
            base_dict(2e-7, 7e-5, 16, 32, 2),
            base_dict(2e-7, 7e-5, 16, 64, 2),
            base_dict(2e-7, 7e-5, 32, 32, 2),
            base_dict(2e-7, 4e-5, 32, 64, 2),
            base_dict(2e-7, 8e-5, 16, 64, 1)
        ],
        4: [
            base_dict(2e-7, 1e-4, 16, 32, 1),
            base_dict(2e-7, 1e-4, 32, 32, 2),
            base_dict(2e-7, 2e-4, 16, 64, 1),
            base_dict(1e-7, 3e-5, 32, 64, 2),
            base_dict(1e-7, 2e-5, 32, 64, 1),
            base_dict(1.5e-7, 3e-5, 16, 64, 2),
            base_dict(2e-7, 2e-5, 16, 32, 2),
            base_dict(2e-7, 2e-5, 32, 32, 1)
        ],
        5: [
            base_dict(1e-7, 2e-5, 16, 64, 1),
            base_dict(1e-7, 3e-5, 16, 64, 2),
            base_dict(2e-7, 2e-5, 16, 32, 1),
            base_dict(2e-7, 3e-5, 16, 32, 2),
            base_dict(2e-7, 7e-5, 32, 32, 1),
            base_dict(2e-7, 2e-5, 32, 64, 2),
            base_dict(2e-7, 2e-5, 32, 64, 1),
            base_dict(2e-7, 3e-5, 32, 32, 2)
        ]
    }
    return dictionary


def return_dictionary_all_weighted(base_dict):
    dictionary = {
        3: [base_dict(5e-7, 5e-6, 32, 32, 1),
            base_dict(1e-6, 2e-5, 32, 32, 2),
            base_dict(1e-6, 5e-6, 32, 64, 2),
            base_dict(1e-6, 5e-6, 32, 64, 1),
            base_dict(1e-6, 5e-6, 16, 64, 2),
            base_dict(1e-6, 5e-6, 16, 64, 1),
            base_dict(5e-7, 5e-6, 16, 32, 1),
            base_dict(1e-6, 2e-5, 16, 32, 2),
            ]
        ,
        4: [base_dict(1e-6, 2e-5, 16, 32, 1),
            base_dict(1e-6, 2e-5, 16, 32, 2),
            base_dict(1e-6, 2e-5, 32, 32, 1),
            base_dict(1e-6, 2e-5, 32, 32, 2),
            base_dict(5e-7, 5e-6, 32, 64, 2),
            base_dict(1e-6, 5e-6, 32, 64, 1),
            base_dict(5e-7, 5e-6, 16, 64, 2),
            base_dict(1e-6, 5e-6, 16, 64, 1)
            ],
        5: [base_dict(1.5e-7, 4e-5, 32, 64, 1)]
    }
    return dictionary


def return_dictionary(base_dict):
    dictionary = {
        1: [
            base_dict(2e-7, 7e-4, 8, 16, 2),
        ],
        2: [
            base_dict(2e-7, 7e-4, 8, 16),
        ],
        3: [
            base_dict(2e-7, 7e-4, 8, 16),
        ],
        4: [
            base_dict(2e-7, 7e-4, 8, 16),
        ],
        5: [
            base_dict(2e-7, 7e-4, 8, 16),
        ],
        6: [
            base_dict(2e-7, 7e-4, 8, 16),
        ],
        7: [
            base_dict(2e-7, 7e-4, 8, 16),
        ],
        8: [
            base_dict(2e-7, 7e-4, 8, 16),
        ]
    }
    return dictionary

def return_dictionary_new(base_dict):
    dictionary = {
        4: [
            base_dict(2e-7, 8e-4, 8, 16, 1),
        ]
    }
    return dictionary


def run_model(gpu=1,min_lr=1e-4, max_lr=1e-2, layers_dict=None, epochs=1000,validation_generator=None,step_size=None,paths_class=None,
              step_size_factor=5, train_generator=None, batch_norm=False,mask_pred=False,pre_cycle=0,write_images=True,
              morfeus_drive='',base_path='', save_a_model=True,weighted=False, mask_loss=False,balance_beta=1.0, model_params=None,**kwargs):
    if step_size is None:
        step_size = len(train_generator)
    G = get_available_gpus()
    if len(G) == 1:
        gpu = 0
    with tf.device('/gpu:' + str(gpu)):
        gpu_options = tf.GPUOptions(allow_growth=True)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        K.set_session(sess)
        if not os.path.exists(morfeus_drive):
            print('Morf wrong')
            return None
        if not os.path.exists(base_path):
            print('base wrong')
            return None


        model_path_out = paths_class.model_path_out
        tensorboard_output = paths_class.tensorboard_path_out

        epoch_i = 0
        optimizer = Adam(lr=min_lr)
        period = 5
        monitor = 'val_loss' #dice_coef_3D
        mode = 'min'
        checkpoint = ModelCheckpoint_new(model_path_out, monitor=monitor, verbose=1, save_best_only=True,
                                         save_weights_only=False, period=period, mode=mode)
        tensorboard = TensorBoardImage(log_dir=tensorboard_output, batch_size=1, write_graph=True, write_grads=False,num_images=3,
                                       update_freq='epoch',  data_generator=validation_generator, image_frequency=3, write_images=write_images)
        lrate = CyclicLR(base_lr=min_lr, max_lr=max_lr, step_size=step_size * step_size_factor, mode='triangular2', pre_cycle=pre_cycle)
        early_stopping = EarlyStopping_BMA(monitor=monitor,min_delta=0,patience=5,verbose=1,mode=mode,
                                           max_delta=1.0,baseline=2.2,restore_best_weights=False)
        early_stopping = EarlyStopping(monitor=monitor, patience=15, verbose=1, mode=mode)
        callbacks = [early_stopping, lrate, tensorboard]
        if save_a_model:
            callbacks = [checkpoint] + callbacks
        loss = 'categorical_crossentropy'
        if weighted:
            loss = weighted_categorical_crossentropy(np.asarray([1,500])) #categorical_crossentropy
            print('weighted loss')
        if mask_loss:
            loss = categorical_crossentropy_masked()
        if balance_beta != 1.0:
            loss = balanced_cross_entropy(balance_beta)
        model = my_3D_UNet(kernel=(3, 3, 3), layers_dict=layers_dict, pool_size=(2, 2, 2),custom_loss=loss,batch_norm=batch_norm,
                           pool_type='Max',out_classes=2,mask_loss=mask_loss, mask_output=mask_pred,**model_params)
        # if return_mask:
        #     loss = weighted_categorical_crossentropy(np.load(os.path.join('.', 'new_class_weights.npy')))
        Model_val = model.created_model
        if mask_loss:
            loss = model.custom_loss
        train = True
        if train:
            Model_val.compile(optimizer, loss=loss, metrics=['accuracy', dice_coef_3D])
            x,y = train_generator.__getitem__(0)
            # pred = Model_val.predict(x)
            Model_val.fit_generator(generator=train_generator, workers=10, use_multiprocessing=False, max_queue_size=50,
                                    shuffle=True, epochs=epochs, callbacks=callbacks, initial_epoch=epoch_i,
                                    validation_data=validation_generator,steps_per_epoch=len(train_generator))


def train_model():
    mask_image = False
    mask_loss = False
    mask_pred = True
    batch_norm = False
    write_images = True
    save_a_model = True
    inverse_images = False
    norm_to_liver = True
    smoothing = 0.0
    weighted = False
    threshold_mask = -7
    if inverse_images:
        threshold_mask = 7
    base_path, morfeus_drive, train_generator, validation_generator = return_generators(inverse_images=inverse_images, liver_norm=norm_to_liver)
    pre_cycle = 0
    gpu = 3
    step_size_factor = 8
    num_cycles = 5
    step_size = len(train_generator)

    base_dict = lambda min_lr, max_lr, filters, max_filters, atrous_rate: \
        OrderedDict({'Architecture':{'model_name':'','layers': 0,'atrous_blocks': 1,'atrous_rate':atrous_rate, 'max_atrous_blocks':1,
                                     'filters':filters, 'max_filters':max_filters,'layers_conv_blocks': 0,
                                     'conv_blocks': 0},
                     'Hyper_Parameters':{'min_lr':min_lr,'max_lr':max_lr,'step_size_factor': step_size_factor,
                                         'num_cycles': num_cycles, 'pre_cycle': pre_cycle}
                     })
    epochs = step_size_factor * 2 * num_cycles
    model_params = {'activation':'relu', 'concat_not_add':False}
    model_name = '3D_Atrous'  # change this
    if norm_to_liver:
        model_name += '_livernorm'
    if inverse_images:
        model_name += '_inversed'
    if weighted:
        model_name += '_weighted'
    if smoothing > 0:
        model_name += '{}_smoothing'.format(smoothing)
    for iteration in [0,1,2]:
        overall_dictionary = return_dictionary(base_dict)
        for layer in overall_dictionary:
            data = overall_dictionary[layer]
            for run_data in data:
                run_data['Architecture']['model_name'] = model_name
                run_data['Architecture']['layers'] = layer
                run_data['Architecture']['batch_norm'] = batch_norm
                run_data['Architecture']['mask_image'] = mask_image
                run_data['Architecture']['mask_pred'] = mask_pred
                run_data['Architecture']['mask_loss'] = mask_loss
                things = return_things(run_data)
                things = things[1:] + ['{}_Iteration'.format(iteration)]
                layers_dict = get_layers_dict_atrous(**run_data['Architecture'])
                # layers_dict = get_layers_dict_conv(layers=layer, **run_data) # change this
                train_generator_3D = Image_Clipping_and_Padding(layers_dict, train_generator, return_mask=mask_pred or mask_loss,
                                                                liver_box=True, mask_image=mask_image,
                                                                remove_liver_layer=True, threshold_value=threshold_mask)
                # x,y = train_generator_3D.__getitem__(0)
                validation_generator_3D = Image_Clipping_and_Padding(layers_dict, validation_generator,
                                                                     threshold_value=threshold_mask,
                                                                     return_mask=mask_pred or mask_loss,liver_box=True,
                                                                     mask_image=mask_image, remove_liver_layer=True)
                # while True:
                #     for i in range(5):
                #         x,y = validation_generator_3D.__getitem__(i)
                paths_class = Path_Return_Class(base_path=base_path, morfeus_path=morfeus_drive, save_model=save_model)
                paths_class.define_model_things(model_name, things)
                tensorboard_output = paths_class.tensorboard_path_out
                # my_3D_UNet(kernel=(3, 3, 3), layers_dict=layers_dict, pool_size=(2, 2, 2), custom_loss=None,
                #            batch_norm=batch_norm,
                #            pool_type='Max', out_classes=2, mask_loss=mask_loss, mask_output=mask_pred,
                #            **model_params)
                if os.listdir(tensorboard_output):
                    continue
                print(tensorboard_output)
                try:
                    run_model(gpu=gpu, layers_dict=layers_dict, train_generator=train_generator_3D, step_size=step_size,
                              validation_generator=validation_generator_3D,save_a_model=save_a_model,model_params=model_params,
                              paths_class=paths_class,morfeus_drive=morfeus_drive, base_path=base_path,
                              epochs=epochs, weighted=weighted, write_images=write_images,**run_data['Architecture'])
                    K.clear_session()
                except:
                    print('failed here')
                    K.clear_session()


if __name__ == '__main__':
    train_model()
