__author__ = 'Brian M Anderson'
# Created on 9/2/2020
import sys, os

sys.path.append('..')
if len(sys.argv) > 1:
    gpu = int(sys.argv[1])
else:
    gpu = 0
print('Running on {}'.format(gpu))
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)


def main():
    from Return_Train_Validation_Generators_TF2 import return_paths
    base_path, morfeus_drive = return_paths()
    from Return_Train_Validation_Generators_TF2 import get_layers_dict_dense_less_decode, my_UNet, return_generators, \
        get_layers_dict_dense_HNet, return_model, plot_scroll_Image, os
    import numpy as np
    import SimpleITK as sitk
    add = '_16'
    cache_add = ''
    model_name = 'DenseNetNew'
    evaluate = False
    if evaluate:
        path = os.path.join(base_path, 'Predictions_np')
        images = [i for i in os.listdir(path) if i.startswith('Image')]
        compare = sitk.LabelOverlapMeasuresImageFilter()
        total = []
        for file in images:
            print(file)
            x = np.load(os.path.join(path, file))
            pred = np.load(os.path.join(path, file.replace('Image', 'Prediction')))
            truth = np.load(os.path.join(path, file.replace('Image', 'Truth')))
            compare.Execute(sitk.GetImageFromArray((pred[..., -1] >= 0.5).astype('int')),
                            sitk.GetImageFromArray(truth.astype('int')))
            total.append(compare.GetDiceCoefficient())
            print(total)
        print(np.mean(total))
        print(np.std(total))
    make_pred = True
    if make_pred:
        base_path, morfeus_drive, train_generator, validation_generator = return_generators(batch_size=0, add=add,
                                                                                            cache_add=cache_add,
                                                                                            flip=True,
                                                                                            change_background=False,
                                                                                            threshold=True,
                                                                                            threshold_val=10,
                                                                                            path_lead='Records',
                                                                                            validation_name='',
                                                                                            cache=False, wanted_keys={
                'inputs': ['image', 'mask', 'image_path'], 'outputs': ['annotation']})
        layers_dict = get_layers_dict_dense_HNet(layers=3, max_conv_blocks=12, filters=32, num_conv_blocks=4, conv_lambda=4)
        model_path = os.path.join(base_path, 'Keras', model_name, 'Models', 'Trial_ID_4', 'final_model.h5')
        model = return_model(layers_dict=layers_dict, densenet=True, all_trainable=True, weights_path=model_path)
        model.save(os.path.join(base_path, 'Keras', model_name, 'Models', 'Trial_ID_4', 'Model'))
        generator = validation_generator.data_set.as_numpy_iterator()
        if not os.path.exists(os.path.join(base_path, 'Predictions_np')):
            os.makedirs(os.path.join(base_path, 'Predictions_np'))
        for i in range(len(validation_generator)):
            print(i)
            x, y = next(generator)
            x = x[:-1]
            file_name = os.path.split(x[-1].decode())[-1]
            pred = model.predict(x)
            np.save(os.path.join(base_path, 'Predictions_np', 'Prediction_{}.npy'.format(file_name)), pred)
            np.save(os.path.join(base_path, 'Predictions_np', 'Image_{}.npy'.format(file_name)), np.squeeze(x[0].numpy()))
            np.save(os.path.join(base_path, 'Predictions_np', 'Truth_{}.npy'.format(file_name)), np.squeeze(y[0].numpy()))

    out_path = os.path.join(base_path, 'Predictions_HNet_1', '{}')
    create_prediction = False
    if create_prediction:
        from Deep_Learning.Evaluate_Model.Write_Predictions import create_prediction_files
        validation_path = [r'H:\Liver_Disease_Ablation\Records\Validation_whole_Records']
        create_prediction_files(is_test=False, desc=desc, model_path=model_path, path_ext=path_ext,
                                out_path=out_path, validation_path=validation_path)
        validation_path = [r'H:\Liver_Disease_Ablation\Records\Test_Records']
        create_prediction_files(is_test=True, desc=desc, model_path=model_path, path_ext=path_ext,
                                out_path=out_path, validation_path=validation_path)

    evaluate_prediction = False
    if evaluate_prediction:
        from Deep_Learning.Evaluate_Model.Evaluate_On_Data_TF2 import create_metric_chart, np, cpu_count
        path = os.path.join(out_path.format(path_ext), 'Validation{}'.format(desc))
        create_metric_chart(path=path, out_path=os.path.join('.', 'Evaluate_Model', 'Threshold_Seed_Pickles_93'),
                            seed_range=np.arange(0.3, 1.0, 0.01),
                            threshold_range=np.arange(0.05, .76, 0.01), re_write=False, thread_count=int(cpu_count()*.9-1))

    evaluate_test = False
    if evaluate_test:
        from Deep_Learning.Evaluate_Model.Evaluate_On_Data_TF2 import create_metric_chart, np
        path = os.path.join(out_path.format(path_ext), 'Test{}'.format(desc))
        # create_metric_chart(path=path, out_path=os.path.join('.', 'Evaluate_Model', 'Test_Output_93'),
        #                     seed_range=[.67], write_final_prediction=True, single_disease=True,
        #                     threshold_range=[.3], re_write=False, thread_count=12)
        create_metric_chart(path=path, out_path=os.path.join('.', 'Evaluate_Model', 'Test_Output_New_Whole_Patient_93'),
                            seed_range=[.67], write_final_prediction=True, single_disease=False,
                            threshold_range=[.3], re_write=False, thread_count=12)

    write_sensitivity_specificity = False
    if write_sensitivity_specificity:
        from Deep_Learning.Evaluate_Model.Sensitivity_and_Specificity_Measures import write_sensitivity_specificity
        write_sensitivity_specificity(excel_path=os.path.join('.', 'Evaluate_Model', 'Sensitivity_and_FP.xlsx'))

    write_box_plots = False
    if write_box_plots:
        from Deep_Learning.Evaluate_Model.Make_Box_Plots import create_plot
        import pandas as pd
        import numpy as np
        out_path = os.path.join('.', 'Test_Output','Final_Prediction.xlsx')
        df = pd.read_excel(out_path, engine='xlrd')
        volumes = df['volume'].values
        values = df['dice'].values
        values = values[volumes < 1]
        values = values[values < 1000]
        create_plot('Dice for less than 1 cc volumes',values=values,metric='Dice', out_path=os.path.join('.','Images'),
                    y_ticks=[0,.1,.2,.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
        xxx = 1


if __name__ == '__main__':
    main()
