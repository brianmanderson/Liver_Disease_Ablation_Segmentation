__author__ = 'Brian M Anderson'
# Created on 1/15/2020

from Plot_And_Scroll_Images.Plot_Scroll_Images import plot_scroll_Image
from LiTs_Into_Niftii import create_NIFTI_images
from Make_Single_Images.Make_Single_Images_Class import run_main


def run_LiTs_to_NIFTII():
    data_path = r'K:\Morfeus\BMAnderson\CNN\Data\Data_Liver\Fuentes_Data\LiTs\Images'
    out_path = r'K:\Morfeus\BMAnderson\CNN\Data\Data_Liver\Liver_Disease_Ablation_Segmentation\Niftii_Data'
    images_desc = 'LiTs'
    create_NIFTI_images(data_path, out_path, images_desc)

def main():
    create_niftii_images = False
    if create_niftii_images:
        run_LiTs_to_NIFTII()
    make_single_images = True
    if make_single_images:
        path = r'K:\Morfeus\BMAnderson\CNN\Data\Data_Liver\Liver_Disease_Ablation_Segmentation\Niftii_Data'
        desired_output_spacing = (0.89648, 0.89648, 3)
        extension = 16
        write_images = True
        re_write_pickle = False
        thread_count = 1
        run_main(path,desired_output_spacing,extension, write_images, re_write_pickle, thread_count)

if __name__ == '__main__':
    main()