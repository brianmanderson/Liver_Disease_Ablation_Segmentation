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

find_lr = True
if find_lr:
    from Optimization.Find_Best_LR import find_best_lr
    find_best_lr(path_extension='Single_Images3D_1mm', cube_size = (30,300,300))
'''
Plot the LR, get the min and max from the images
'''
plot_lr = False
if plot_lr:
    from Optimization.Plot_Best_LR import make_plots
    from Return_Train_Validation_Generators import return_generators
    _, morfeus_drive, _, _ = return_generators()
    path = os.path.join(morfeus_drive,'3.25_Learning_Rates','Fully_Atrous')
    make_plots(path)

'''
Now, we need to run the model for a number of epochs ~200, so we can get a nice curve to make final model
decision based on
'''
run_200 = False
if run_200:
    from Run_Model import train_model
    # iterate_model(epochs=100, save_a_model=True, runv3_plus=False, run_vgg=False)
    train_model(epochs=100, save_a_model=True, run_best=False, ext='_1mm')
    # iterate_model(epochs=100, save_a_model=True, runv3_plus=False, run_vgg=True, retrain=True)

make_opt_excel = False
if make_opt_excel:
    '''
    Need to run the model for ~ 200 epochs, then run Plot_Optimization_results
    '''
    from Optimization.Plot_Optimization_results import main
    main()

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
    train_model(run_best=True,save_a_model=True)