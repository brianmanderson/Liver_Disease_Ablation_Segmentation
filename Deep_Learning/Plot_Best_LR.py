from Base_Deeplearning_Code.Finding_Optimization_Parameters.LR_Finder import make_plot, os


def down_folder(input_path, base_input_path=''):
    folders = []
    for _, folders, _ in os.walk(input_path):
        break
    iterations_found = False
    for folder in folders:
        if folder.find('Iteration') == -1:
            down_folder(os.path.join(input_path, folder),base_input_path=base_input_path)
        else:
            iterations_found = True
            break
    if iterations_found:
        paths = [os.path.join(input_path, i) for i in folders if i.find('Iteration') != -1]
        try:
            print(input_path)
            desc = ''.join(input_path.split(base_input_path)[-1].split('\\'))
            save_path = os.path.join(base_input_path,'Outputs')
            make_plot(paths, metric_list=['loss'], title=desc, save_path=save_path, plot=True,
                      auto_rates=True, beta=0.98)
        except:
            xxx = 1
    return None

out_path = os.path.join('..','..','Learning_Rates_Liver_Disease')
down_folder(out_path,base_input_path=out_path)