__author__ = 'Brian M Anderson'
# Created on 9/28/2020

'''
This gets broken down into a couple of steps... 
1) we have to unzip the data
2) turn the dicom masks into actual RT structures
3) Can upload them now into RS for viewing, or not
4) Create predictions of disease
5) Evaluate differences between predictions and ground truth
'''

'''
1) Unzip Data
'''
unzip = False
if unzip:
    from Unzip_Tool.Unzip_Tool import Unzip_class, os
    path = r'H:\Liver_Disease_Ablation\3Dircadb1\Downloaded_Patients'
    if not os.path.exists(os.path.join(path, 'Unzipped')):
        Unzip_class(path=path, file='3Dircadb1.zip')

'''
2) Turn dicom masks into actual RT structures
'''
turn_dicom_to_RS = False
data_path = r'H:\Liver_Disease_Ablation\3Dircadb1\Downloaded_Patients\Unzipped\3Dircadb1'
dicom_path = r'H:\Liver_Disease_Ablation\3Dircadb1\Fixed_Patients'
if turn_dicom_to_RS:
    from Create_Ground_Truth import create_dicom_RT
    create_dicom_RT(path=data_path, out_path=dicom_path)

'''
3) Create disease predictions
'''
make_disease_predictions = False
if make_disease_predictions:
    from Create_Ground_Truth import create_predictions
    prediction_path = r'L:\Clinical\Auto_Contour_Sites\Liver_Disease_Ablation_Auto_Contour\Input_3'
    prediction_path = r'H:\AutoModels\Liver_Disease\Input_5'
    create_predictions(prediction_path=prediction_path, image_path=dicom_path)

'''
4) Copy the outcoming RS structure locally
'''
copy_predictions_locally = False
if copy_predictions_locally:
    from Create_Ground_Truth import copy_predictions
    prediction_out_path = r'L:\Clinical\Auto_Contour_Sites\Liver_Disease_Ablation_Auto_Contour\Output'
    copy_predictions(prediction_out_path=prediction_out_path, image_path=dicom_path)

'''
5) Compare predictions with the ground truth
'''
compare_with_gt = False
if compare_with_gt:
    from Create_Ground_Truth import compare_predictions, os
    compare_predictions(path=dicom_path, out_path=r'H:\Liver_Disease_Ablation\3Dircadb1\Niftii_Files',
                        excel_path=os.path.join('.', 'Patient_Results.xlsx'))


single_site = True
if single_site:
    from Sensitivity_and_Specificity_Measures import single_site_comparison, pd, os
    out_dict = single_site_comparison(path=r'H:\Liver_Disease_Ablation\3Dircadb1\Niftii_Files')
    df = pd.DataFrame(out_dict)
    df.to_excel(os.path.join('.', 'Single_Site_Results.xlsx'), index=0)



write_sensitivity_specificity = False
if write_sensitivity_specificity:
    from Sensitivity_and_Specificity_Measures import write_sensitivity_specificity, os
    write_sensitivity_specificity(excel_path=os.path.join('.', 'Sensitivity_and_FP.xlsx'),
                                  nifti_path=r'H:\Liver_Disease_Ablation\3Dircadb1\Niftii_Files')