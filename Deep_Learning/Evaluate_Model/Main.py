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


model_name = 'weights-improvement-best_cube.hdf5'
create_prediction = True
desc = 'Cube_Training'
if create_prediction:
    from Deep_Learning.Evaluate_Model.Write_Predictions import create_prediction_files
    create_prediction_files(is_test=False, path_ext='_None', desc=desc, model_name=model_name)

    create_prediction_files(is_test=True, path_ext='_None', desc=desc, model_name=model_name)

evaluate_prediction = False
if evaluate_prediction:
    from Deep_Learning.Evaluate_Model.Iterate_Predictions import create_metric_chart, os
    create_metric_chart(desc=desc,metric_range = [.1,.2,.3,.4,.5,.6,.7,.8,.9],out_path=os.path.join('.','Threshold_Expand'))

evaluate_test = False
if evaluate_test:
    from Deep_Learning.Evaluate_Model.Iterate_Predictions import create_metric_chart, os
    create_metric_chart(desc=desc,metric_range = [.3],
                        path = r'D:\Liver_Disease_Ablation\Predictions_1mm\TestFWHM',
                        out_path=os.path.join('.','Test_Output'))