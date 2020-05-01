__author__ = 'Brian M Anderson'
# Created on 4/26/2020
from Base_Deeplearning_Code.Data_Generators.Return_Paths import *
import tensorflow as tf
from Base_Deeplearning_Code.Plot_And_Scroll_Images.Plot_Scroll_Images import plot_scroll_Image
from Base_Deeplearning_Code.Finding_Optimization_Parameters.LR_Finder import LearningRateFinder
from Return_Train_Validation_Generators_TF2 import return_generators, return_base_dict
from Base_Deeplearning_Code.Models.TF_Keras_Models import my_UNet


def return_things(run_data):
    things = []
    for top_key in ['Architecture','Hyper_Parameters']:
        model_info = run_data[top_key]
        for key in model_info:
            if model_info[key] is not False:
                if model_info[key] is True:
                    things.append('{}'.format(key))
                else:
                    things.append('{}_{}'.format(model_info[key],key))
    return things


def find_best_lr(optimizer='SGD'):
    base_dict = return_base_dict(optimizer=optimizer)
    min_lr = 1e-7
    max_lr = 1e-2
    for iteration in [0, 1, 2]:
        for layer in [1,2,3,4,5]: #
            for filters in [8, 16]: #, 16
                for max_filters in [16, 32]: #, 32
                    run_data = base_dict(min_lr=min_lr, max_lr=max_lr, filters=filters, max_filters=max_filters,
                                         layers=layer)
                    layers_dict = get_layers_dict_atrous(**run_data['Architecture'])
                    things = return_things(run_data)
                    things.append('{}_Iteration'.format(iteration))
                    out_path = os.path.join(morfeus_drive,path_desc,'Fully_Atrous')
                    for thing in things:
                        out_path = os.path.join(out_path,thing)
                    if os.path.exists(out_path):
                        continue
                    print(out_path)
                    os.makedirs(out_path)
                    try:
                        run_model(layers_dict=layers_dict, out_path=out_path,
                                  train_generator=train_generator)
                        K.clear_session()
                    except:
                        K.clear_session()


if __name__ == '__main__':
    pass
