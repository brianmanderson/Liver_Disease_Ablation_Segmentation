import numpy as np
from matplotlib import pyplot
import os, glob
import nibabel as nib
import scipy.misc as scm
import matplotlib.pyplot as plt
from threading import Thread
from multiprocessing import cpu_count
from queue import *


def worker_def(A):
    q, base_path, out_path, one_rotation, dont_flip_lr = A
    while True:
        item = q.get()
        if item is None:
            break
        else:
            try:
                write_image(base_path,item,out_path,one_rotation)
            except:
                print('failed?')
            q.task_done()

def write_image(base_path,image_file,out_path,one_rotation):
    print(image_file)
    path = 'C:\\users\\bmanderson\\desktop\\images_LiTs\\'
    index = (image_file.split('volume-')[1]).split('.nii')[0]
    if os.path.exists(os.path.join(out_path, 'Overall_mask_LiTs_y' + str(index))):
        return None
    if index in one_rotation:
        num_turns = 1
    else:
        num_turns = 3
    label = nib.load(os.path.join(base_path, 'segmentation-' + index + '.nii'))
    label = label._data
    if np.max(label) == 1:
        return None
    img = nib.load(os.path.join(base_path, image_file))
    img = img._data
    for i in range(num_turns):
        img = np.rot90(img)
    for i in range(num_turns):
        label = np.rot90(label)
    # If there is more on the right side than left side, flip it
    if np.sum(label[...,:256,:]==1) < np.sum(label[...,256:,:]==1):
        img = np.fliplr(img)
        label = np.fliplr(label)
    locations = np.where(label == 1)[2]
    i = (locations.max() - locations.min()) // 2 + locations.min()
    out = np.zeros([512, 512 * 2])
    out[:, 0:512] = img[:, :, i]
    # out[out>300] = 300
    # out[out<-100] = -100
    max_val = img[:, :, i].max()
    out[:, 512:] = label[:, :, i] * max_val
    scm.imsave(os.path.join(path, 'combined_' + str(index) + '.png'), out)
    np.save(os.path.join(out_path, 'Overall_Data_LiTs_' + str(index)+'.npy'), img)
    np.save(os.path.join(out_path, 'Overall_mask_LiTs_y' + str(index)+'.npy'), label)
    return None
img_dir = 'C:\\CNN\\Fuentes_Data\\Fuentes_Liver\\Batch_2\\media\\nas\\01_Datasets\\CT\\LITS\\Training Batch 2\\'
img_dir = '\\\\mymdafiles\\di_data1\\Morfeus\\BMAnderson\\CNN\\Data\\Data_Liver\\Fuentes_Data\\LiTs\\Images\\'
out_path = '\\\\mymdafiles\\di_data1\\Morfeus\\BMAnderson\\CNN\\Data\\Data_Liver\\Fuentes_Data\\Abdomen\\RawData\\Training\\Numpy\\'
out_path = '\\\\mymdafiles\\di_data1\\Morfeus\\BMAnderson\\CNN\\Data\\Data_Liver\\Fuentes_Data\\LiTs\\Numpy\\All_Data\\'
one_rotation = ['70','46','30','51','76','78','24','44','37','32','69','31','41','8','81','15','79','18','0','26','47','14',
          '25','34','74','68','50','28','36','5','12','1','77','48','7','23','29','21','115','11','13','38','42',
          '82','73','10','17','39','4','20','33','19','43','35','27','6','71','3','9','16','75','45','22','80','49',
          '52','72','2','40']
three_rotation = ['89','59','123','67','95','86','108','129','118','66','124','103','56','54','105','58','97','130','98','113',
          '111','53','85','61','63','121','84','99','92','88','110','91','112','65','62','107','126','83','90','120',
          '127','119','114','64','109','128','125','101','100','116','60','102','57','122','55','93','96','87','106',
          '94']
dont_flip_lr = ['24','30','0','8','15','18','24','26','30','31','37','44','46','51']

path = 'C:\\users\\bmanderson\\desktop\\images_LiTs\\'
if not os.path.exists(path):
    os.makedirs(path)
images_dir = os.listdir(img_dir)
xxx = 1
num_turns = 3
file_list = [i for i in os.listdir(img_dir) if i.find('volume') != -1]

thread_count = cpu_count() - 1  # Leaves you one thread for doing things with
# thread_count = 1
print('This is running on ' + str(thread_count) + ' threads')
q = Queue(maxsize=thread_count)
A = [q, img_dir, out_path, one_rotation, dont_flip_lr]
threads = []
for worker in range(thread_count):
    t = Thread(target=worker_def, args=(A,))
    t.start()
    threads.append(t)
for file in file_list:
    q.put(file)
for i in range(thread_count):
    q.put(None)
for t in threads:
    t.join()
# for file in file_list:
#     write_image(img_dir,file,out_path,one_rotation, dont_flip_lr)
#     continue