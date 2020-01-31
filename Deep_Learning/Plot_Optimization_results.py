__author__ = 'Brian M Anderson'
# Created on 1/30/2020
from Base_Deeplearning_Code.Finding_Optimization_Parameters.History_Plotter import create_excel_from_event, \
    plot_from_excel, os, np, partial, pd
import plotnine as p9
from plotnine import *


make_excel = False
input_path = r'K:\Morfeus\BMAnderson\CNN\Data\Data_Liver\Liver_Disease_Ablation_Segmentation\Keras\3D_Atrous_livernorm'
if make_excel:
    create_excel_from_event(input_path=input_path,names=['Layers','Filters','Max_Filters','Atrous_Blocks','atrous_rate','max_atrous_blocks','min_lr','max_lr','step_size_factor','num_cycles'])

excel_out_path = os.path.join('.', 'Model_Optimization.xlsx')
data = pd.read_excel(excel_out_path)
data = data.dropna()
(ggplot(data) + aes(x='layers',y='val_loss') + geom_point(mapping=aes(color='max_filters')) + facet_wrap('filters')
 + xlab('Layers') + ylab('Val_Loss') + ggtitle('Validation Loss vs parameters')+scale_color_gradient(low='blue',high='red'))
xxx = 1
# criteria_base = lambda x, variable_name, value: np.asarray(list(x[variable_name].values())) == value
# criteria = partial(criteria_base, variable_name='max_filters', value=32)
# plot_from_excel(excel_out_path, variable_name='layers') #,criterias=[criteria]