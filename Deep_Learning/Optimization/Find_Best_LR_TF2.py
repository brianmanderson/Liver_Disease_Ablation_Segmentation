__author__ = 'Brian M Anderson'
# Created on 4/26/2020
from Base_Deeplearning_Code.Data_Generators.Return_Paths import *
import tensorflow as tf
from Base_Deeplearning_Code.Plot_And_Scroll_Images.Plot_Scroll_Images import plot_scroll_Image
from Base_Deeplearning_Code.Finding_Optimization_Parameters.LR_Finder import LearningRateFinder
from Return_Train_Validation_Generators_TF2 import return_generators, return_base_dict, get_layers_dict, return_hyper_parameters
from Base_Deeplearning_Code.Callbacks.TF2_Callbacks import SparseCategoricalMeanDSC
from Base_Deeplearning_Code.Models.TF_Keras_Models import my_UNet


def find_best_lr(optimizer='SGD', batch_size=16, path_desc='', bn_before_activation=True):
    base_dict = return_base_dict(optimizer=optimizer)
    if optimizer == 'SGD':
        min_lr = 1e-7
        max_lr = 1
    else:
        min_lr = 1e-10
        max_lr = 1
    hp_dict = return_hyper_parameters()
    for iteration in [0]:
        for layer in hp_dict['layers'].domain.values:
            for filters in hp_dict['filters'].domain.values:
                for max_filters in hp_dict['max_filters'].domain.values:
                    base_path, morfeus_drive, train_generator, validation_generator = return_generators(
                        batch_size=batch_size)
                    run_data = base_dict(min_lr=min_lr, max_lr=max_lr, filters=filters, max_filters=max_filters,
                                         layers=layer)
                    layers_dict = get_layers_dict(**run_data, bn_before_activation=bn_before_activation)
                    things = ['layers_{}'.format(layer), 'filters_{}'.format(filters),
                              'max_filters_{}'.format(max_filters), 'Optimizer_{}'.format(optimizer)]
                    things.append('{}_Iteration'.format(iteration))
                    out_path = os.path.join(morfeus_drive,path_desc,'Fully_Atrous')
                    for thing in things:
                        out_path = os.path.join(out_path,thing)
                    if os.path.exists(out_path):
                        continue
                    os.makedirs(out_path)
                    print(out_path)
                    model = my_UNet(layers_dict=layers_dict, image_size=(None, None, None, 1),
                                    mask_output=True).created_model
                    # optimizer = tf.keras.optimizers.Adam()
                    # optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(optimizer)
                    # model.compile(optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                    #               metrics=[tf.keras.metrics.SparseCategoricalAccuracy(),
                    #                        SparseCategoricalMeanDSC(num_classes=2)])
                    # model.fit(train_generator.data_set, epochs=5, steps_per_epoch=20,
                    #           validation_data=validation_generator.data_set, validation_steps=5)
                    # data = next(iter(validation_generator.data_set))
                    # pred = model(data[0])
                    if optimizer == 'SGD':
                        lr_opt = tf.keras.optimizers.SGD
                    else:
                        lr_opt = tf.keras.optimizers.Adam
                    LearningRateFinder(epochs=10, model=model, metrics=['sparse_categorical_accuracy'],
                                       out_path=out_path, optimizer=lr_opt,
                                       loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                                       steps_per_epoch=len(train_generator),
                                       train_generator=train_generator.data_set, lower_lr=min_lr, high_lr=max_lr)
                    tf.keras.backend.clear_session()


if __name__ == '__main__':
    pass
