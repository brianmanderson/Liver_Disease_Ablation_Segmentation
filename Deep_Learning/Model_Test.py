__author__ = 'Brian M Anderson'
# Created on 5/29/2020

from Return_Train_Validation_Generators_TF2 import get_layers_dict_dense_less_decode, my_UNet, return_generators
from Run_Model_TF2 import SparseCategoricalMeanDSC
from tensorflow.keras.callbacks import TensorBoard
import tensorflow.keras as tfk
from HDenseUNet.denseunet import DenseUNet
from MyHybridDenseNet.Loading_Pretrained_DenseNet import DenseNet121

add = ''
cache_add = ''
base_path, morfeus_drive, train_generator, validation_generator = return_generators(
    batch_size=0, add='_16', threshold_val=10, change_background=False,
    cache_add=cache_add, path_lead='Records', validation_name='_64')
# x,y = next(iter(train_generator.data_set))
# x1,y1 = next(iter(validation_generator.data_set))
model = DenseNet121(include_top=False, classes=2)
trainable = False
for index, layer in enumerate(model.layers):
    if layer.name.find('Upsampling') == 0:
        trainable = True
    model.layers[index].trainable = trainable
model.compile(tfk.optimizers.Adam(),
              loss=tfk.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=[tfk.metrics.SparseCategoricalAccuracy(), SparseCategoricalMeanDSC(num_classes=2)])
model.fit(train_generator.data_set, epochs=10,steps_per_epoch=len(train_generator),
          validation_data=validation_generator.data_set, validation_steps=len(validation_generator),
          validation_freq=1)
xxx = 1
model = DenseUNet(reduction=0.5)
model.load_weights(r'H:\Liver_Disease_Ablation\densenet161_weights_tf.h5', by_name=True)
out_path = r'D:\test_model\test'
layers_dict = get_layers_dict_dense_less_decode(layers=4, filters=8, growth_rate=0, num_conv_blocks=2, conv_lambda=0)
model = my_UNet(layers_dict=layers_dict, image_size=(None, None, None, 1),
                mask_output=True).created_model
k = TensorBoard(log_dir=out_path, profile_batch=0, write_graph=True)
k.set_model(model)
k.on_train_begin()