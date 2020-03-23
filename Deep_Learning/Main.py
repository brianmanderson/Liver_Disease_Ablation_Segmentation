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


'''
Now, we need to run the model for a number of epochs ~200, so we can get a nice curve to make final model
decision based on
'''
run_200 = True
if run_200:
    from Run_Model import train_model
    train_model(run_best=True,save_a_model=True)