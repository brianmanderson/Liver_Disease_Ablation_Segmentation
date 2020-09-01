__author__ = 'Brian M Anderson'
# Created on 8/31/2020

import sys, os

if len(sys.argv) > 1:
    gpu = int(sys.argv[1])
else:
    gpu = 0
print('Running on {}'.format(gpu))
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'

cube_size = (16, 120, 120)
add = '_16'
path_desc='TF_LR_2D_Dense_1mm_new'
model_name = 'DenseNet'
cache_add = ''
from Return_Train_Validation_Generators_TF2 import return_paths
base_path, morfeus_drive = return_paths()
kernel = (3, 3)
squeeze_kernel = (1, 1)
find_dense_lr_dense = False
if find_dense_lr_dense:
    from Optimization.Find_Best_LR_TF2_Dense import find_best_lr
    find_best_lr(batch_size=0, path_desc=path_desc, add=add, cache_add=cache_add, kernel=kernel,
                 squeeze_kernel=squeeze_kernel, image_size=(None, None, 1))

find_dense_lr_densenet121 = False
model_path = os.path.join(base_path, 'Keras', 'DenseNet', 'Models', 'Trial_ID_2', 'cp-best.h5')
if find_dense_lr_densenet121:
    from Optimization.Find_Best_LR_TF2_Dense import find_best_lr_DenseNet
    find_best_lr_DenseNet(batch_size=0, path_desc=path_desc, add=add, cache_add=cache_add, path_lead='Records',
                          all_trainable=True, weights_path=model_path)

'''
Plot the LR, get the min and max from the images
'''
plot_lr = False
if plot_lr:
    from Optimization.Plot_Best_LR import make_plots
    path = os.path.join(morfeus_drive,path_desc, 'DenseNet121')
    make_plots(path)

'''
Now, we need to run the model for a number of epochs ~200, so we can get a nice curve to make final model
decision based on
'''
run_200 = True
if run_200:
    from Run_Model_TF2 import train_DenseNet
    run_best = False
    train_DenseNet(epochs=201, model_name=model_name, run_best=run_best, add=add,  cache_add=cache_add, batch_size=0,
                   change_background=False, path_lead='Records', validation_name='_64', all_trainable=True,
                   weights_path=model_path)

make_opt_excel = False
if make_opt_excel:
    '''
    Need to run the model for ~ 200 epochs, then run Plot_Optimization_results
    '''
    from Optimization.Plot_Optimization_results_TF2 import main
    from Return_Train_Validation_Generators_TF2 import return_paths
    base_path, morfeus_drive = return_paths()
    main()
