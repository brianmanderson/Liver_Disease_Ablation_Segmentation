__author__ = 'Brian M Anderson'
# Created on 3/2/2020

from Deep_Learning.Evaluate_Model.Write_Predictions import create_prediction_files


def main():
    # First, create all prediction files from the validation set (not test set yet, we need to find the best cutoff)
    create_prediction_files(is_test=False, path_ext='_1mm')

    create_prediction_files(is_test=True, path_ext='_1mm')



if __name__ == '__main__':
    main()

