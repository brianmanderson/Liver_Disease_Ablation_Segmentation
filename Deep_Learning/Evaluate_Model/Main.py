__author__ = 'Brian M Anderson'
# Created on 3/2/2020



create_prediction = False
if create_prediction:
    from Deep_Learning.Evaluate_Model.Write_Predictions import create_prediction_files
    desc = 'FWHM'
    create_prediction_files(is_test=False, path_ext='_1mm', desc=desc)

    create_prediction_files(is_test=True, path_ext='_1mm', desc=desc)

evaluate_prediction = False
if evaluate_prediction:
    from Deep_Learning.Evaluate_Model.Iterate_Predictions import create_metric_chart, os
    create_metric_chart(desc='FWHM',metric_range = [.1,.2,.3,.4,.5,.6,.7,.8,.9],out_path=os.path.join('.','Threshold_Expand'))

evaluate_test = True
if evaluate_test:
    from Deep_Learning.Evaluate_Model.Iterate_Predictions import create_metric_chart, os
    create_metric_chart(desc='FWHM',metric_range = [.3],
                        path = r'D:\Liver_Disease_Ablation\Predictions_1mm\TestFWHM',
                        out_path=os.path.join('.','Test_Output'))