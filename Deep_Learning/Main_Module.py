__author__ = 'Brian M Anderson'
# Created on 1/17/2020

from Return_Train_Validation_Generators import return_generators
from Base_Deeplearning_Code.Plot_And_Scroll_Images.Plot_Scroll_Images import plot_scroll_Image

train_generator, validation_generator = return_generators()
x,y = train_generator.__getitem__(0)
x = 1
