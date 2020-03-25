import sys
sys.path.append('..')
from Base_Deeplearning_Code.Data_Generators.Image_Processors import *
import tensorflow.python.keras.backend as K
import tensorflow.compat.v1 as tf
from Base_Deeplearning_Code.Models.Keras_Models import my_UNet
from Base_Deeplearning_Code.Keras_Utils.Keras_Utilities import dice_coef_3D
from Base_Deeplearning_Code.Finding_Optimization_Parameters.LR_Finder import LearningRateFinder, Adam
from Return_Train_Validation_Generators import return_generators, return_base_dict, get_layers_dict, OrderedDict


def run_model(layers_dict=None, out_path='',train_generator=None, is_deeplab=False):
    x,y = train_generator.__getitem__(0)
    print(len(train_generator))
    with tf.device('/gpu:0'):
        gpu_options = tf.GPUOptions()
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        tf.keras.backend.set_session(sess)
        loss = 'categorical_crossentropy'
        if layers_dict is not None:
            model = my_UNet(kernel=(3, 3, 3), layers_dict=layers_dict, pool_type='Max', out_classes=2).created_model
            epochs = 10
        else:
            epochs = 5
            if is_deeplab:
                model = Deeplabv3(input_shape=(512, 512, 3), classes=2, backbone='xception', activation='softmax')
            else:
                model = return_vgg_model()
        LearningRateFinder(epochs=epochs, model=model, metrics=['accuracy', dice_coef_3D],out_path=out_path,loss=loss,
                           train_generator=train_generator, lower_lr=1e-7, high_lr=1e-2)
        K.clear_session()


def return_things(run_data):
    things = []
    for top_key in ['Architecture']:
        model_info = run_data[top_key]
        for key in model_info:
            if model_info[key] is not False:
                if model_info[key] is True:
                    things.append('{}'.format(key))
                else:
                    things.append('{}_{}'.format(model_info[key],key))
    return things


def find_best_lr():
    base_path, morfeus_drive, train_generator, validation_generator = return_generators()
    x,y = train_generator.__getitem__(0)
    base_dict = return_base_dict()
    min_lr = 1e-7
    max_lr = 1e-2
    for iteration in [0, 1, 2]:
        for layer in [2,3,4]:
            for conv_layers in [2, 3]:
                for num_atrous_blocks in [2]:
                    for atrous_rate in [2]:
                        for filters in [8]:
                            for max_filters in [32]:
                                run_data = base_dict(layer,conv_layers,num_atrous_blocks,atrous_rate,filters,max_filters, min_lr, max_lr)
                                layers_dict = get_layers_dict(**run_data['Architecture'])
                                things = return_things(run_data)
                                things.append('{}_Iteration'.format(iteration))
                                out_path = os.path.join(morfeus_drive,'Learning_Rates_Liver_3D')
                                for thing in things:
                                    out_path = os.path.join(out_path,thing)
                                if os.path.exists(out_path):
                                    continue
                                print(out_path)
                                os.makedirs(out_path)
                                try:
                                    run_model(layers_dict=layers_dict, out_path=out_path,
                                              train_generator=train_generator)
                                except:
                                    K.clear_session()


def find_best_lr_2D(is_deeplab=True):
    base_path, morfeus_drive, train_generator, validation_generator = return_generators(is_3D=False)
    x,y = train_generator.__getitem__(0)
    run_data = return_base_dict(is_v3plus=is_deeplab, is_vgg=not is_deeplab)
    run_data = run_data(1e-7,1e-2)
    out_name = 'v3_plus'
    if not is_deeplab:
        out_name = 'vgg_16'
    for iteration in [0, 1, 2]:
        things = return_things(run_data)
        things.append('{}_Iteration'.format(iteration))
        out_path = os.path.join(morfeus_drive,'Learning_Rates_{}'.format(out_name))
        for thing in things:
            out_path = os.path.join(out_path,thing)
        if os.path.exists(out_path):
            continue
        print(out_path)
        os.makedirs(out_path)
        try:
            run_model(out_path=out_path, train_generator=train_generator, is_deeplab=is_deeplab)
        except:
            K.clear_session()


if __name__ == '__main__':
    pass


