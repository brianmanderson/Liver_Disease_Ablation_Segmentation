__author__ = 'Brian M Anderson'
# Created on 1/27/2020
from Base_Deeplearning_Code.Visualizing_Model.Visualing_Model import visualization_model_class
from Base_Deeplearning_Code.Keras_Utils.Keras_Utilities import dice_coef_3D
from Return_Train_Validation_Generators import return_generators
from Base_Deeplearning_Code.Data_Generators.Generators import Image_Clipping_and_Padding, np
from keras.models import load_model
from Base_Deeplearning_Code.Plot_And_Scroll_Images.Plot_Scroll_Images import plot_scroll_Image
import os


def get_layers_dict(layers=1, filters=16, conv_blocks=1, num_atrous_blocks=4, max_blocks=2, max_filters=np.inf,
                    atrous_rate=1, max_atrous_rate=2, **kwargs):
    activation = 'relu'
    layers_dict = {}
    conv_block = lambda x: {'convolution': {'channels': x, 'kernel': (3, 3, 3), 'strides': (1, 1, 1),'activation':activation}}
    strided_block = lambda x: {'convolution': {'channels': x, 'kernel': (3, 3, 3), 'strides': (2, 2, 2), 'activation':activation}}
    transpose_block = lambda x: {'transpose': {'channels': x, 'kernel': (3, 3, 3), 'strides': (2, 2, 2), 'activation':'linear'}}
    atrous_block = lambda x,y,z: {'atrous': {'channels': x, 'atrous_rate': y, 'activations': z}}
    for layer in range(conv_blocks,layers-1):
        encoding = [atrous_block(filters,atrous_rate,[activation for _ in range(atrous_rate)]) for _ in range(num_atrous_blocks)]
        atrous_block_dec = [atrous_block(filters,atrous_rate,[activation for _ in range(atrous_rate)]) for _ in range(num_atrous_blocks)]
        if layer == 0:
            encoding = [conv_block(filters)] + encoding
        dec_tranpose = transpose_block(filters)
        if filters < max_filters:
            filters = int(filters*2)
        layers_dict['Layer_' + str(layer)] = {'Encoding': encoding,
                                              'Pooling':{'Encoding':[strided_block(filters)],'Decoding':[dec_tranpose]},
                                              'Decoding': atrous_block_dec}
        num_atrous_blocks = min([(num_atrous_blocks) * 2,max_blocks])
    num_atrous_blocks = min([(num_atrous_blocks) * 2, max_blocks])
    layers_dict['Base'] = {'Encoding':[atrous_block(filters,atrous_rate,[activation for _ in range(atrous_rate)]) for _ in range(num_atrous_blocks)]}
    return layers_dict


def return_dictionary(base_dict):
    dictionary = {
        4: [
            base_dict(2e-7, 5e-4, 8, 16, 1)
            # base_dict(2e-7, 7e-4, 8, 32, 1)
        ],
        5: [
            base_dict(2e-7, 5e-4, 8, 16, 1),
            base_dict(2e-7, 7e-4, 8, 32, 1)
        ]
    }
    return dictionary


def return_val_generator():
    mask_image = False
    mask_loss = False
    mask_pred = True
    batch_norm = False
    write_images = True
    save_a_model = True
    inverse_images = True
    norm_to_liver = True
    smoothing = 0.0
    weighted = False
    threshold_mask = -7
    if inverse_images:
        threshold_mask = 7
    base_path, morfeus_drive, train_generator, validation_generator = return_generators(inverse_images=inverse_images,
                                                                                        liver_norm=norm_to_liver)
    step_size_factor = 5
    num_cycles = 8
    base_things = {'num_conv_blocks': 2, 'conv_blocks': 0, 'num_convs': 2, 'num_atrous_blocks': 1,
                   'step_size_factor': step_size_factor, 'num_cycles': num_cycles, 'pre_cycle': 0,
                   'atrous_rate': 2, 'max_atrous_rate': 2}
    base_dict = lambda a, b, c, d, e: {'min_lr': a, 'max_lr': b, 'filters': c, 'max_filters': d, 'max_blocks': e}
    overall_dictionary = return_dictionary(base_dict)
    layer = 4
    data = overall_dictionary[layer]
    for run_data in data:
        base_things['batch_norm'] = batch_norm
        base_things['mask_image'] = mask_image
        base_things['smoothing'] = smoothing
        base_things['mask_pred'] = mask_pred
        base_things['write_images'] = write_images
        base_things['mask_loss'] = mask_loss
        run_data.update(base_things)  # Change this
        run_data['Layers'] = str(layer)
        layers_dict = get_layers_dict(layers=layer, **run_data)
        # layers_dict = get_layers_dict_conv(layers=layer, **run_data) # change this
        validation_generator_3D = Image_Clipping_and_Padding(layers_dict, validation_generator,
                                                             threshold_value=threshold_mask,
                                                             return_mask=mask_pred or mask_loss, liver_box=True,
                                                             mask_image=mask_image, remove_liver_layer=True)
        return validation_generator_3D

model_path = r'\\mymdafiles\di_data1\Morfeus\BMAnderson\CNN\liver_disease_model\weights-improvement-best.hdf5'
out_path = r'\\mymdafiles\di_data1\Morfeus\bmanderson\Modular_Projects\Liver_Disease_Segmentation_Work'
model = load_model(model_path, custom_objects={'dice_coef_3D':dice_coef_3D})
Visualizing_Class = visualization_model_class(model=model, save_images=True,out_path=os.path.join(out_path,'Activations'))
validation_generator = return_val_generator()
x,y = validation_generator.__getitem__(0)
Visualizing_Class.predict_on_tensor(x)
Visualizing_Class.plot_activations()