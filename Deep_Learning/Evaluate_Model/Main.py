__author__ = 'Brian M Anderson'
# Created on 3/2/2020

import sys, os

if len(sys.argv) > 1:
    gpu = int(sys.argv[1])
else:
    gpu = 0
print('Running on {}'.format(gpu))
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
desc = 'TF2_Multi_Cube'
model_path = r'D:\Liver_Disease_Ablation\Keras\TF2_3D_Fully_Atrous_Variable_Cube_Training\Models\Trial_ID_199\model'

create_model = False
if create_model:
    weight_path = r'D:\Liver_Disease_Ablation\Keras\TF2_3D_Fully_Atrous_Variable_Cube_Training\Models\Trial_ID_199\cp-0260.ckpt'
    from Return_Train_Validation_Generators_TF2 import return_base_dict, get_layers_dict_new
    from Base_Deeplearning_Code.Models.TF_Keras_Models import my_UNet
    base_dict = return_base_dict()
    run_data = base_dict(max_conv_blocks=4, layers=2, num_conv_blocks=3, conv_lambda=1, filters=32, max_filters=128,
                         min_lr=4e-7, max_lr=1.75e-3)
    layers_dict = get_layers_dict_new(**run_data)
    model = my_UNet(layers_dict=layers_dict, image_size=(None, None, None, 1), mask_output=True, concat_not_add=True).created_model
    model.load_weights(weight_path)
    model.save(model_path)

create_prediction = False
if create_prediction:
    from Deep_Learning.Evaluate_Model.Write_Predictions import create_prediction_files
    create_prediction_files(is_test=False, desc=desc, model_path=model_path)

    create_prediction_files(is_test=True, desc=desc, model_path=model_path)

evaluate_prediction = False
if evaluate_prediction:
    from Deep_Learning.Evaluate_Model.Evaluate_On_Data_TF2 import create_metric_chart, os, np
    path = r'D:\Liver_Disease_Ablation\Predictions\ValidationTF2_Multi_Cube'
    create_metric_chart(path=path,desc=desc,out_path=os.path.join('.','Threshold_Seed_Pickles'),
                        seed_range=[.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75,.8,.85,.9,.95,.975,.99],
                        threshold_range=np.arange(0.1,0.96,0.01), re_write=False, thread_count=12)

evaluate_test = True
if evaluate_test:
    from Deep_Learning.Evaluate_Model.Evaluate_On_Data_TF2 import create_metric_chart, os
    path = r'D:\Liver_Disease_Ablation\Predictions\TestTF2_Multi_Cube'
    create_metric_chart(path=path,desc=desc,out_path=os.path.join('.','Test_Output'),
                        seed_range=[.99], threshold_range=[.35], re_write=False,
                        write_final_prediction=True, thread_count=10)