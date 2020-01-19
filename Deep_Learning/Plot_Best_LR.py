from Base_Deeplearning_Code.Finding_Optimization_Parameters.LR_Finder import make_plot, os

out_path = os.path.join('..','..','Learning_Rates_Liver_Disease_GTV_weighted200')
all_lrs, all_metrics = [], []
path = out_path
for _, folders, _ in os.walk(path):
    break
for i in folders:
    paths = os.path.join(path, i)
    iterations = []
    for _, iterations, _ in os.walk(paths):
        break
    if iterations:
        paths = [os.path.join(path,i,iii) for iii in iterations]
    print(i)
    try:
        make_plot(paths,metric_list=['loss'],title=i, save_path=os.path.join(out_path), plot=True, auto_rates=True)

    except:
        continue