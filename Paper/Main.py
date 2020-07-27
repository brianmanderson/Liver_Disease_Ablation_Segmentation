__author__ = 'Brian M Anderson'
# Created on 7/27/2020

combine_excel_sheets = False
if combine_excel_sheets:
    from Paper.Combine_Odisio_Kang_Responses import combine_sheets
    combine_sheets()


find_image_parameters = True
if find_image_parameters:
    from Paper.Create_Image_Parameters_CSV import identify_image_parameters
    identify_image_parameters(images_path=r'K:\Morfeus\BMAnderson\CNN\Data\Data_Liver'
                                          r'\Liver_Disease_Ablation_Segmentation\Nifti_to_dicom\LiTs',
                              xlsx_path=r'K:\Morfeus\BMAnderson\Modular_Projects'
                                        r'\Liver_Disease_Ablation_Segmentation_Work\Image_parameters_LiTs.xlsx')
