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
path_desc='TF2_Learning_Rates'
model_name = 'TF2_3D_Fully_Atrous_Variable_Cube_Training'
find_lr = False
if find_lr:
    from Optimization.Find_Best_LR_TF2 import find_best_lr
    bn_before_activation = False
    find_best_lr(optimizer='SGD', batch_size=16, path_desc=path_desc, bn_before_activation=bn_before_activation)
'''
Plot the LR, get the min and max from the images
'''
plot_lr = False
if plot_lr:
    from Optimization.Plot_Best_LR import make_plots
    from Return_Train_Validation_Generators_TF2 import return_generators
    _, morfeus_drive, _, _ = return_generators()
    path = os.path.join(morfeus_drive,path_desc)
    make_plots(path)

'''
Now, we need to run the model for a number of epochs ~200, so we can get a nice curve to make final model
decision based on
'''
run_200 = True
if run_200:
    from Run_Model_TF2 import train_model
    optimizer = 'SGD'
    bn_before_activation = True
    train_model(epochs=50, save_a_model=False, bn_before_activation=bn_before_activation, model_name=model_name, optimizer=optimizer)

make_opt_excel = False
if make_opt_excel:
    '''
    Need to run the model for ~ 200 epochs, then run Plot_Optimization_results
    '''
    from Optimization.Plot_Optimization_results import main
    from Return_Train_Validation_Generators import return_generators
    _, morfeus_drive, _, _ = return_generators(path_extension=path_extension)
    main(make_excel=True, input_path= os.path.join(morfeus_drive,'Keras',model_name))

'''
Now go to Evaluate_Model folder
'''

'''
Now, we need to run the model for a number of epochs ~200, so we can get a nice curve to make final model
decision based on
'''
run_200 = False
if run_200:
    from Run_Model import train_model
    sgd=False
    for step_size_factor in [20]:
        for add in [10]:
            train_model(epochs=1005, step_size_factor=step_size_factor,
                        save_a_model=True, run_best=True, path_extension=path_extension,
                        cube_size=cube_size, model_name=model_name, step_size_add=add, sgd=sgd)