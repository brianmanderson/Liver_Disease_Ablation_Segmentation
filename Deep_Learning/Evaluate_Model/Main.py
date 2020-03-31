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


model_name = 'weights-improvement-best_cube_600.hdf5'
desc = 'Cube_Training'
create_prediction = True
if create_prediction:
    from Deep_Learning.Evaluate_Model.Write_Predictions import create_prediction_files
    create_prediction_files(is_test=False, path_ext='_None', desc=desc, model_name=model_name)

    create_prediction_files(is_test=True, path_ext='_None', desc=desc, model_name=model_name)

evaluate_prediction = False
if evaluate_prediction:
    from Deep_Learning.Evaluate_Model.Evaluate_On_Data import create_metric_chart, os
    path = r'D:\Liver_Disease_Ablation\Predictions_None\Validation{}'.format(desc)
    create_metric_chart(path=path,desc=desc,out_path=os.path.join('.','Threshold_Seed'),
                        threshold_range = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7])

evaluate_test = False
if evaluate_test:
    from Deep_Learning.Evaluate_Model.Evaluate_On_Data import create_metric_chart, os
    create_metric_chart(desc=desc,threshold_range = [.4],
                        path = r'D:\Liver_Disease_Ablation\Predictions_None\Test{}'.format(desc),
                        out_path=os.path.join('.','Test_Output'))