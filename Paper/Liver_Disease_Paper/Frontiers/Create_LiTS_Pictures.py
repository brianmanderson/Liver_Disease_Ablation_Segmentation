import SimpleITK as sitk
import numpy as np
import os
from glob import glob
from PIL import Image
from PlotScrollNumpyArrays.Plot_Scroll_Images import plot_scroll_Image
from tqdm import tqdm

path = r'K:\LiTs\Training_Batch2\media\nas\01_Datasets\CT\LITS\Training Batch 2'
volume_files = glob(os.path.join(path, "vol*.nii"))
prog = tqdm(total=len(volume_files), desc='Creating jpegs')
for file in volume_files:
    segmentation_handle = sitk.ReadImage(file.replace('volume', 'segmentation'))
    segmentation_array = sitk.GetArrayFromImage(segmentation_handle)
    liver_mask = np.where(np.max(segmentation_array, axis=(1, 2)) > 0)
    image_handle = sitk.ReadImage(file)
    image_array = sitk.GetArrayFromImage(image_handle).astype('float32')
    image_array[image_array<-100] = -100
    image_array[image_array>200] = 200
    image_array += 100
    image_array *= 255/300
    image = Image.fromarray(image_array[int(np.median(liver_mask[0]))].astype('uint8'))
    out_path_jpeg = file.replace('.nii', '.jpeg')
    image.save(out_path_jpeg)
    prog.update()