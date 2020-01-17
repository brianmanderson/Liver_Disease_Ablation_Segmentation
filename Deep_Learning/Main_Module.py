__author__ = 'Brian M Anderson'
# Created on 1/17/2020

from Return_Train_Validation_Generators import return_generators
from Base_Deeplearning_Code.Data_Generators.Generators import Image_Clipping_and_Padding
from Base_Deeplearning_Code.Plot_And_Scroll_Images.Plot_Scroll_Images import plot_scroll_Image
import numpy as np


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


mask_image = True
mask_pred = False
batch_norm = False
write_images = False
train_generator, validation_generator = return_generators()
base_things = {'num_conv_blocks': 2, 'conv_blocks': 1, 'num_convs': 2, 'num_atrous_blocks': 1,
               'step_size_factor': 5, 'num_cycles': 5}
base_things['batch_norm'] = batch_norm
base_things['mask_image'] = mask_image
base_things['mask_pred'] = mask_pred
base_things['write_images'] = write_images
base_dict = lambda a, b, c, d, e: {'min_lr': a, 'max_lr': b, 'filters': c, 'max_filters': d, 'max_blocks': e}
run_data = base_dict(4e-6, 5.7e-4, 32, 64, 1)
x,y = train_generator.__getitem__(0)
layers_dict = get_layers_dict(layers=3, **run_data)
train_generator_3D = Image_Clipping_and_Padding(layers_dict,train_generator,return_mask=mask_pred,liver_box=True,
                                                mask_image=mask_image, threshold_value=3.55, remove_liver_layer=True)
x,y = train_generator_3D.__getitem__(0)
x = 1
