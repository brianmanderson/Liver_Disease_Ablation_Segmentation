plot_lr = False
path_desc='TF2_Learning_Rates'
import os
if plot_lr:
    from Optimization.Plot_Best_LR import make_plots
    from Return_Train_Validation_Generators_TF2 import return_generators
    _, morfeus_drive, _, _ = return_generators()
    path = os.path.join(morfeus_drive,path_desc)
    make_plots(path)