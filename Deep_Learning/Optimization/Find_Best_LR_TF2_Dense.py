__author__ = 'Brian M Anderson'
# Created on 4/26/2020
from Base_Deeplearning_Code.Data_Generators.Return_Paths import *
import tensorflow as tf
from Base_Deeplearning_Code.Plot_And_Scroll_Images.Plot_Scroll_Images import plot_scroll_Image
from Base_Deeplearning_Code.Finding_Optimization_Parameters.LR_Finder import LearningRateFinder
from tensorflow.keras.callbacks import TensorBoard
from Return_Train_Validation_Generators_TF2 import return_generators, return_base_dict, get_layers_dict_dense, return_paths
from Base_Deeplearning_Code.Models.TF_Keras_Models import my_UNet


def find_best_lr(batch_size=16, path_desc='', add=''):
    min_lr = 1e-7
    max_lr = 1
    for iteration in [0]:
        for growth_rate in [4]:
            for layer in [2, 3]:
                for max_conv_blocks in [4]:
                    for filters in [8, 12]:
                        for num_conv_blocks in [2]:
                            for conv_lambda in [0, 1]:
                                base_path, morfeus_drive = return_paths()
                                run_data = {'layers':layer,'max_conv_blocks':max_conv_blocks,'filters':filters,
                                            'num_conv_blocks':num_conv_blocks, 'conv_lambda':conv_lambda,
                                            'growth_rate':growth_rate}
                                layers_dict = get_layers_dict_dense(**run_data)
                                things = ['layers{}'.format(layer), 'max_conv_blocks_{}'.format(max_conv_blocks),
                                          'filters_{}'.format(filters), 'num_conv_blocks_{}'.format(num_conv_blocks),
                                          'conv_lambda_{}'.format(conv_lambda), 'growth_rate_{}'.format(growth_rate),
                                          '{}_Iteration'.format(iteration)]
                                out_path = os.path.join(morfeus_drive,path_desc,'Dense')
                                for thing in things:
                                    out_path = os.path.join(out_path,thing)
                                if os.path.exists(out_path):
                                    print('already done')
                                    continue
                                os.makedirs(out_path)
                                print(out_path)
                                base_path, morfeus_drive, train_generator, validation_generator = return_generators(
                                    batch_size=batch_size, add=add, threshold_val=10, change_background=True)
                                model = my_UNet(layers_dict=layers_dict, image_size=(None, None, None, 1),
                                                mask_output=True).created_model
                                k = TensorBoard(log_dir=out_path, profile_batch=0, write_graph=True)
                                k.set_model(model)
                                k.on_train_begin()
                                lr_opt = tf.keras.optimizers.Adam
                                LearningRateFinder(epochs=10, model=model, metrics=['sparse_categorical_accuracy'],
                                                   out_path=out_path, optimizer=lr_opt,
                                                   loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                                                   steps_per_epoch=len(train_generator),
                                                   train_generator=train_generator.data_set, lower_lr=min_lr, high_lr=max_lr)
                                tf.keras.backend.clear_session()
                                return None # repeat!


if __name__ == '__main__':
    pass
