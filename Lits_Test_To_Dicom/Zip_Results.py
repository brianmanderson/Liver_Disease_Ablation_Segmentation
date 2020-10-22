__author__ = 'Brian M Anderson'
# Created on 9/25/2020
from zipfile import ZipFile, ZIP_DEFLATED
import os


def get_all_file_paths(directory):
    # initializing empty file paths list
    file_paths = []

    # crawling through directory and subdirectories
    for root, directories, files in os.walk(directory):
        for filename in files:
            # join the two strings in order to form the full filepath.
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)

            # returning all file paths
    return file_paths


def zip_files():
    # path to folder which needs to be zipped
    path = r'H:\Liver_Disease_Ablation\LiTs_Test\Nifti'
    out_file = os.path.join(path, 'Last_Submission_bma.zip')
    # calling function to get all file paths in the directory
    file_paths = [os.path.join(path, i) for i in os.listdir(path) if i.startswith('test-segmentation-')]

    # printing the list of all files to be zipped
    # print('Following files will be zipped:')
    # for file_name in file_paths:
    #     print(file_name)

    # writing files to a zipfile
    with ZipFile(out_file, 'w', ZIP_DEFLATED) as zip:
        # writing each file one by one
        for file in file_paths:
            print(file)
            zip.write(file)

    print('All files zipped successfully!')


if __name__ == '__main__':
    pass
