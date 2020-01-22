__author__ = 'Brian M Anderson'
# Created on 1/18/2020
from Base_Deeplearning_Code.Data_Generators.Generators import Image_Clipping_and_Padding
from Base_Deeplearning_Code.Data_Generators.Return_Paths import *
from keras.models import *
import tensorflow as tf
from Base_Deeplearning_Code.Keras_Utils.Keras_Utilities import dice_coef_3D_np, ModelCheckpoint_new, get_available_gpus, save_obj,load_obj, \
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


def return_things(run_data):
    middle_info = ''
    if run_data['batch_norm']:
        middle_info += '_BatchNorm'
    if run_data['mask_image']:
        middle_info += '_MaskedImage'
    if run_data['mask_pred']:
        middle_info += '_MaskedPred'
    things = ['{}_Layers'.format(run_data['Layers']),
              '{}_Convblocks'.format(run_data['conv_blocks']),
              '{}_Max_Atrous'.format(run_data['max_blocks']),
              '{}_Filters_{}_Max_Filters'.format(run_data['filters'],run_data['max_filters']),
              middle_info,
              '{}_MinLR_{}_MaxLR'.format(run_data['min_lr'], run_data['max_lr']),
              '{}_step_size_{}_precycles_{}_cycles'.format(run_data['step_size_factor'], run_data['pre_cycle'],
                                                           run_data['num_cycles']),
              'Iteration_{}'.format(run_data['Iteration'])]
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
        4: [
            base_dict(2e-7, 1e-4, 16, 32, 1)
        ]
    }
    return dictionary


def run_model(gpu=1,min_lr=1e-4, max_lr=1e-2, layers_dict=None, epochs=1000,validation_generator=None,step_size=None,paths_class=None,
              step_size_factor=5, train_generator=None, batch_norm=False,mask_pred=False,pre_cycle=0,write_images=True,
              morfeus_drive='',base_path='', save_a_model=True,weighted=False, mask_loss=False, **kwargs):
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
        monitor = 'val_dice_coef_3D'
        mode = 'max'
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
        model = my_3D_UNet(filter_vals=(3, 3, 3), layers_dict=layers_dict, pool_size=(2, 2, 2),custom_loss=loss,batch_norm=batch_norm,
                            activation='elu', pool_type='Max',out_classes=2,mask_loss=mask_loss, mask_output=mask_pred)
        # if return_mask:
        #     loss = weighted_categorical_crossentropy(np.load(os.path.join('.', 'new_class_weights.npy')))
        Model_val = model.created_model
        if mask_loss:
            loss = model.custom_loss
        train = True
        if train:
            Model_val.compile(optimizer, loss=loss, metrics=['accuracy', dice_coef_3D])
            # x,y = train_generator.__getitem__(0)
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
    save_a_model = False
    base_path, morfeus_drive, train_generator, validation_generator = return_generators()
    pre_cycle = 0
    gpu = 2
    step_size_factor = 10
    num_cycles = 5
    step_size = len(train_generator)
    base_things = {'num_conv_blocks': 2, 'conv_blocks': 1, 'num_convs': 2, 'num_atrous_blocks': 1,
                   'step_size_factor': step_size_factor, 'num_cycles': num_cycles, 'pre_cycle': pre_cycle}
    base_dict = lambda a, b, c, d, e: {'min_lr': a, 'max_lr': b, 'filters': c, 'max_filters': d, 'max_blocks': e}
    epochs = step_size_factor * 2 * num_cycles
    base_things['batch_norm'] = batch_norm
    base_things['mask_image'] = mask_image
    base_things['mask_pred'] = mask_pred
    base_things['write_images'] = write_images
    base_things['mask_loss'] = mask_loss
    for iteration in range(2):
        for weighted in [False]:
            model_name = '3D_Atrous'  # change this
            if weighted:
                model_name += '_weighted'
            if weighted:
                overall_dictionary = return_dictionary_all_weighted(base_dict)  # change this
            else:
                overall_dictionary = return_dictionary_all(base_dict)
            overall_dictionary = return_dictionary(base_dict)
            for layer in [4]:
                data = overall_dictionary[layer]
                for run_data in data:
                    run_data.update(base_things)  # Change this
                    run_data['Layers'] = str(layer)
                    run_data['Iteration'] = str(iteration)
                    layers_dict = get_layers_dict(layers=layer, **run_data)
                    # layers_dict = get_layers_dict_conv(layers=layer, **run_data) # change this
                    train_generator_3D = Image_Clipping_and_Padding(layers_dict, train_generator, return_mask=mask_pred or mask_loss,
                                                                    liver_box=True, mask_image=mask_image,
                                                                    remove_liver_layer=True, threshold_value=3.55)
                    validation_generator_3D = Image_Clipping_and_Padding(layers_dict, validation_generator,
                                                                         threshold_value=3.55,
                                                                         return_mask=mask_pred or mask_loss,liver_box=True,
                                                                         mask_image=mask_image, remove_liver_layer=True)
                    x,y = train_generator_3D.__getitem__(0)
                    paths_class = Path_Return_Class(base_path=base_path, morfeus_path=morfeus_drive)
                    things = return_things(run_data)
                    paths_class.define_model_things(model_name, things)
                    tensorboard_output = paths_class.tensorboard_path_out
                    if os.listdir(tensorboard_output):
                        continue
                    print(tensorboard_output)
                    try:
                        run_model(gpu=gpu, layers_dict=layers_dict, train_generator=train_generator_3D, step_size=step_size,
                                  validation_generator=validation_generator_3D,save_a_model=save_a_model,
                                  paths_class=paths_class,morfeus_drive=morfeus_drive, base_path=base_path,
                                  epochs=epochs, model_name=model_name, weighted=weighted, **run_data)
                        K.clear_session()
                    except:
                        print('failed here')
                        K.clear_session()

if __name__ == '__main__':
    train_model()