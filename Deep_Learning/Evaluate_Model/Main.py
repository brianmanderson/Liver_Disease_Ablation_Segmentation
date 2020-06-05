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


def main():
    desc = 'TF2_Multi_Cube_1mm'
    model_path = r'D:\Liver_Disease_Ablation\Keras\TF2_3D_Fully_Atrous_Variable_Cube_Training_1mm\Models\Trial_ID_34\model'
    weight_path = r'D:\Liver_Disease_Ablation\Keras\TF2_3D_Fully_Atrous_Variable_Cube_Training_1mm\Models\Trial_ID_34\cp-0201.ckpt'
    dense = True
    path_ext = ''
    if not os.path.exists(model_path) and not dense:
        from Return_Train_Validation_Generators_TF2 import return_base_dict, get_layers_dict_new
        from Base_Deeplearning_Code.Models.TF_Keras_Models import my_UNet
        base_dict = return_base_dict()
        run_data = base_dict(max_conv_blocks=4, layers=2, num_conv_blocks=3, conv_lambda=1, filters=32, max_filters=128,
                             min_lr=4e-7, max_lr=1.75e-3)
        layers_dict = get_layers_dict_new(**run_data)
        model = my_UNet(layers_dict=layers_dict, image_size=(None, None, None, 1), mask_output=True, concat_not_add=True).created_model
        model.load_weights(weight_path)
        model.save(model_path)
    elif dense and not os.path.exists(model_path):
        from Return_Train_Validation_Generators_TF2 import return_base_dict, get_layers_dict_dense
        from Base_Deeplearning_Code.Models.TF_Keras_Models import my_UNet
        layers_dict = get_layers_dict_dense(layers=2, num_conv_blocks=2, max_conv_blocks=4, conv_lambda=0, filters=8,
                                            growth_rate=4)
        model = my_UNet(layers_dict=layers_dict, image_size=(None, None, None, 1), mask_output=True, concat_not_add=True).created_model
        model.load_weights(weight_path)
        model.save(model_path)

    create_prediction = False
    if create_prediction:
        from Deep_Learning.Evaluate_Model.Write_Predictions import create_prediction_files
        validation_path = [r'D:\Liver_Disease_Ablation\Records\Validation_whole_Records']
        create_prediction_files(is_test=False, desc=desc, model_path=model_path, path_ext=path_ext, validation_path=validation_path)
        validation_path = [r'D:\Liver_Disease_Ablation\Records\Test_Records']
        create_prediction_files(is_test=True, desc=desc, model_path=model_path, path_ext=path_ext, validation_path=validation_path)

    evaluate_prediction = False
    if evaluate_prediction:
        from Deep_Learning.Evaluate_Model.Evaluate_On_Data_TF2 import create_metric_chart, np
        path = r'D:\Liver_Disease_Ablation\Predictions{}\Validation{}'.format(path_ext, desc)
        create_metric_chart(path=path,out_path=os.path.join('.','Threshold_Seed_Pickles'),
                            seed_range=np.arange(0.8,1.0,0.01),
                            threshold_range=np.arange(0.1,.6,0.05), re_write=False, thread_count=10)

    evaluate_test = False
    if evaluate_test:
        from Deep_Learning.Evaluate_Model.Evaluate_On_Data_TF2 import create_metric_chart
        path = r'D:\Liver_Disease_Ablation\Predictions{}\Test{}'.format(path_ext, desc)
        create_metric_chart(path=path,out_path=os.path.join('.','Test_Output'),
                            seed_range=[.93], threshold_range=[.2], re_write=False,
                            write_final_prediction=True)

    write_box_plots = True
    if write_box_plots:
        from Deep_Learning.Evaluate_Model.Make_Box_Plots import create_plot
        import pandas as pd
        import numpy as np
        out_path = os.path.join('.', 'Test_Output','Final_Prediction.xlsx')
        df = pd.read_excel(out_path, engine='xlrd')
        volumes = df['volume'].values
        values = df['dice'].values
        values = values[volumes>20]
        values = values[values<1000]
        create_plot('Dice for greater than 20 cc volumes',values=values,metric='Dice', out_path=os.path.join('.','Images'),
                    y_ticks=[0,.1,.2,.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
        xxx = 1


if __name__ == '__main__':
    main()
