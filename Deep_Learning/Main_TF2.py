__author__ = 'Brian M Anderson'
# Created on 3/23/2020

import sys, os

if len(sys.argv) > 1:
    gpu = int(sys.argv[1])
else:
    gpu = 0
print('Running on {}'.format(gpu))
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

path_extension = 'Single_Images3D_None'
cube_size = (16, 16, 120, 120)
batch_size = 16
path_desc='TF2_Learning_Rates_new'
model_name = 'TF2_3D_Fully_Atrous_Variable_Cube_Training'
find_lr = True
if find_lr:
    from Optimization.Find_Best_LR_TF2 import find_best_lr
    find_best_lr(optimizer='Adam', batch_size=16, path_desc=path_desc)
'''
Plot the LR, get the min and max from the images
'''
plot_lr = False
if plot_lr:
    from Optimization.Plot_Best_LR import make_plots
    from Return_Train_Validation_Generators_TF2 import return_paths
    _, morfeus_drive = return_paths()
    path = os.path.join(morfeus_drive,path_desc, 'Fully_Atrous')
    make_plots(path)

'''
Now, we need to run the model for a number of epochs ~200, so we can get a nice curve to make final model
decision based on
'''
run_200 = False
if run_200:
    from Run_Model_TF2 import train_model
    run_best = True
    train_model(epochs=301, model_name=model_name, run_best=run_best, debug=False)

make_opt_excel = False
if make_opt_excel:
    '''
    Need to run the model for ~ 200 epochs, then run Plot_Optimization_results
    '''
    from Optimization.Plot_Optimization_results_TF2 import main
    from Return_Train_Validation_Generators_TF2 import return_paths
    base_path, morfeus_drive = return_paths()
    main()
