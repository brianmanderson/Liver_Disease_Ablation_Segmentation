__author__ = 'Brian M Anderson'
# Created on 7/27/2020

combine_excel_sheets = False
if combine_excel_sheets:
    from Paper.Combine_Odisio_Kang_Responses import combine_sheets
    combine_sheets()


find_image_parameters = False
if find_image_parameters:
    from Paper.Create_Image_Parameters_CSV import identify_image_parameters
    identify_image_parameters(images_path=r'K:\Morfeus\BMAnderson\CNN\Data\Data_Liver'
                                          r'\Liver_Disease_Ablation_Segmentation\Nifti_to_dicom\LiTs',
                              xlsx_path=r'K:\Morfeus\BMAnderson\Modular_Projects'
                                        r'\Liver_Disease_Ablation_Segmentation_Work\Image_parameters_LiTs.xlsx')

combine_parameters_distribution = False
if combine_parameters_distribution:
    import os
    import pandas as pd
    parameters_path = r'K:\Morfeus\BMAnderson\Modular_Projects' \
                      r'\Liver_Disease_Ablation_Segmentation_Work\Image_parameters_LiTs.xlsx'
    data_path = r'H:\Liver_Disease_Ablation'
    paths_dict = {}
    for folder in ['Train', 'Test', 'Validation']:
        files = os.listdir(os.path.join(data_path, folder))
        files = [i.split('Overall_Data_')[1].split('.nii')[0] for i in files if i.startswith('Overall_Data')]
        paths_dict[folder] = files
        xxx = 1
    image_parameters = pd.read_excel(parameters_path)
    out_dict = {'MRN': [], 'Folders': []}
    for mrn in image_parameters['MRN'].values:
        found = False
        for folder in paths_dict.keys():
            if mrn in paths_dict[folder]:
                out_dict['Folders'].append(folder)
                found = True
                break
        if not found:
            out_dict['Folders'].append('NA')
        out_dict['MRN'].append(mrn)
    df = pd.DataFrame(out_dict)
    image_parameters = pd.merge(image_parameters, df, on='MRN')
    parameters_path = r'K:\Morfeus\BMAnderson\Modular_Projects' \
                      r'\Liver_Disease_Ablation_Segmentation_Work\Image_parameters_LiTs.xlsx'
    with pd.ExcelWriter(parameters_path) as writer:
        for folder in ['Train', 'Validation', 'Test', 'NA']:
            image_parameters[image_parameters['Folders'] == folder].to_excel(writer, sheet_name=folder, index=0)
