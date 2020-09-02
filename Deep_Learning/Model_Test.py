__author__ = 'Brian M Anderson'
# Created on 5/29/2020

import sys, os

if len(sys.argv) > 1:
    gpu = int(sys.argv[1])
else:
    gpu = 0
print('Running on {}'.format(gpu))
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'

from Return_Train_Validation_Generators_TF2 import get_layers_dict_dense_less_decode, my_UNet, return_generators, get_layers_dict_dense_HNet, return_model, plot_scroll_Image, os
# from Run_Model_TF2 import SparseCategoricalMeanDSC
import numpy as np
# from tensorflow.keras.callbacks import TensorBoard
# import tensorflow.keras as tfk
# from HDenseUNet.denseunet import DenseUNet
# from MyHybridDenseNet.Loading_Pretrained_DenseNet import DenseNet121

add = '_16'
path_desc='TF_LR_2D_Dense_1mm_new'
model_name = 'DenseNet'
cache_add = ''
# x,y = next(iter(train_generator.data_set))
# x1,y1 = next(iter(validation_generator.data_set))
# path = r'H:\Liver_Disease_Ablation\Predictions_np'
# images = [i for i in os.listdir(path) if i.startswith('Image')]
# for file in images:
#     x = np.load(os.path.join(path, file))
#     pred = np.load(os.path.join(path, file.replace('Image', 'Prediction')))
#     truth = np.load(os.path.join(path, file.replace('Image', 'Truth')))
#     xxx = 1
base_path, morfeus_drive, train_generator, validation_generator = return_generators(batch_size=0, add=add,cache_add=cache_add,
                                                                                    flip=True, change_background=False,
                                                                                    threshold=True, threshold_val=10,
                                                                                    path_lead='Records', validation_name='',
                                                                                    cache=False)
layers_dict = get_layers_dict_dense_HNet(layers=2, filters=32, num_conv_blocks=4, conv_lambda=2)
model_path = os.path.join(base_path, 'Keras', 'DenseNet', 'Models', 'Trial_ID_21', 'final_model.h5')
model = return_model(layers_dict=layers_dict, densenet=True, all_trainable=True, weights_path=model_path)
model.save(os.path.join(base_path, 'Keras', 'DenseNet', 'Models', 'Trial_ID_21', 'Model'))
generator = iter(validation_generator.data_set)
if not os.path.exists(os.path.join(base_path, 'Predictions_np')):
    os.makedirs(os.path.join(base_path, 'Predictions_np'))
for i in range(len(validation_generator)):
    print(i)
    x, y = next(generator)
    pred = model.predict(x)
    np.save(os.path.join(base_path, 'Predictions_np', 'Prediction_{}.npy'.format(i)), pred)
    np.save(os.path.join(base_path, 'Predictions_np', 'Image_{}.npy'.format(i)), np.squeeze(x[0].numpy()))
    np.save(os.path.join(base_path, 'Predictions_np', 'Truth_{}.npy'.format(i)), np.squeeze(y[0].numpy()))
    xxx = 1
# model.compile(tfk.optimizers.Adam(),
#               loss=tfk.losses.SparseCategoricalCrossentropy(from_logits=False),
#               metrics=[tfk.metrics.SparseCategoricalAccuracy(), SparseCategoricalMeanDSC(num_classes=2)])
# model.fit(train_generator.data_set, epochs=10,steps_per_epoch=len(train_generator),
#           validation_data=validation_generator.data_set, validation_steps=len(validation_generator),
#           validation_freq=1)
# xxx = 1
# model = DenseUNet(reduction=0.5)
# model.load_weights(r'H:\Liver_Disease_Ablation\densenet161_weights_tf.h5', by_name=True)
# out_path = r'D:\test_model\test'
# layers_dict = get_layers_dict_dense_less_decode(layers=4, filters=8, growth_rate=0, num_conv_blocks=2, conv_lambda=0)
# model = my_UNet(layers_dict=layers_dict, image_size=(None, None, None, 1),
#                 mask_output=True).created_model
# k = TensorBoard(log_dir=out_path, profile_batch=0, write_graph=True)
# k.set_model(model)
# k.on_train_begin()