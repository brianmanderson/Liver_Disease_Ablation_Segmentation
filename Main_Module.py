__author__ = 'Brian M Anderson'
# Created on 1/15/2020

from Plot_And_Scroll_Images.Plot_Scroll_Images import plot_scroll_Image
from LiTs_Into_Niftii import create_NIFTI_images


def run_LiTs_to_NIFTII():
    data_path = r'K:\Morfeus\BMAnderson\CNN\Data\Data_Liver\Fuentes_Data\LiTs\Images'
    out_path = r'K:\Morfeus\BMAnderson\CNN\Data\Data_Liver\Liver_Disease_Ablation_Segmentation\Niftii_Data'
    images_desc = 'LiTs'
    create_NIFTI_images(data_path, out_path, images_desc)

def main():
    create_niftii_images = True
    if create_niftii_images:
        run_LiTs_to_NIFTII()
    xxx = 1

if __name__ == '__main__':
    main()