__author__ = 'Brian M Anderson'
# Created on 1/30/2020
from Base_Deeplearning_Code.Finding_Optimization_Parameters.History_Plotter import create_excel_from_event, \
    plot_from_excel, os, np, partial


make_excel = True
input_path = r'K:\Morfeus\BMAnderson\CNN\Data\Data_Liver\Liver_Disease_Ablation_Segmentation\Keras\3D_Atrous_strideddown_livernorm'
if make_excel:
    create_excel_from_event(input_path)

excel_out_path = os.path.join('.', 'Model_Optimization.xlsx')
criteria_base = lambda x, variable_name, value: np.asarray(list(x[variable_name].values())) == value
criteria = partial(criteria_base, variable_name='layers', value=5)
plot_from_excel(excel_out_path, variable_name='filters')