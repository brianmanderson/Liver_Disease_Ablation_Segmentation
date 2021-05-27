from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

path = r'.\Margins_and_classes_{}.txt'
for desc, color in zip(['Deformed', 'Rigid'], ['Red', 'Blue']):
    prediction, truth = [], []
    fid = open(path.format(desc))
    for line in fid:
        prediction.append(float(line.split(',')[0]))
        truth.append(1 - float(line.split(',')[1].strip('\n')))
    fid.close()
    fpr, tpr, threshold = roc_curve(truth, prediction)
    roc_auc = auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, color, label='{} AUC: %0.2f'.format(desc) % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
plt.show()
xxx = 1