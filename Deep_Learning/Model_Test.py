__author__ = 'Brian M Anderson'
# Created on 5/29/2020

from Return_Train_Validation_Generators_TF2 import get_layers_dict_dense_new, my_UNet
from tensorflow.keras.callbacks import TensorBoard

out_path = r'D:\test_model\test'
layers_dict = get_layers_dict_dense_new(layers=2, filters=8, growth_rate=0, num_conv_blocks=2, conv_lambda=1)
model = my_UNet(layers_dict=layers_dict, image_size=(None, None, None, 1),
                mask_output=True).created_model
k = TensorBoard(log_dir=out_path, profile_batch=0, write_graph=True)
k.set_model(model)
k.on_train_begin()