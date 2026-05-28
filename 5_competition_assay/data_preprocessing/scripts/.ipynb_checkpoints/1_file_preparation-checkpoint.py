import tifffile
import os
from skimage import io, color, filters, morphology
from PIL import Image
import numpy as np
import tifffile as tiff
from tqdm import tqdm 
import matplotlib.pyplot as plt

def otsu_thresholding(input_folder,output_folder) :
    os.makedirs(output_folder, exist_ok=True)
    fichiers = [f for f in os.listdir(input_folder) if f.endswith('.tif')]
    for f in fichiers:
        with tifffile.TiffFile(os.path.join(input_folder, f)) as tif:
            image_stack = tif.asarray()  
            # Save each channel
            #for i in range(1,(image_stack.shape[0])): #not keeping channel 1=brightfield
            for i in range(0,(image_stack.shape[0])): #if no brightfield
                thresh = filters.threshold_otsu(image_stack[i])
                binary = image_stack[i] > thresh
                output_path = os.path.join(output_folder, f'C{i+1}-'+ f)
                tifffile.imwrite(output_path, image_stack[i]*binary)

### INPUT USER ###--
# Path to the input multi-channel TIFF
input_path = r"/Users/laureleblanc/Desktop/Collective-twitching-2026/5_competition_assay/data_preprocessing/1_Raw_images"
output_path = r"/Users/laureleblanc/Desktop/Collective-twitching-2026/5_competition_assay/data_preprocessing/2_Thresholded"
###-----------------
otsu_thresholding(input_path, output_path)

def remove_small_objects(chemin_image, dossier_sortie, seuil_aire, threshold_intensity, visualize=False):
    '''
    Is necessary on some images where otsu is not good enough to threshold them.
    Removes small objects whose area is smaller than seuil_aire, and you can manually put an intensity threshold to replace Otsu if it was not strong enough
    '''
    for nom_fichier in os.listdir(chemin_image):
        if nom_fichier.lower().endswith(".tif") or nom_fichier.lower().endswith(".tiff"):
            source=os.path.join(chemin_image, nom_fichier)
            image = io.imread(source)
            imagebool = image > threshold_intensity  
            masque_filtré = morphology.remove_small_objects(imagebool.astype(bool), min_size=seuil_aire)
            res = image * masque_filtré
            chemin_sortie = os.path.join(dossier_sortie, 'copied'+nom_fichier)
            io.imsave(chemin_sortie, res)
    

# remove_small_objects(input_path,output_path, 5, 1597)


#Annex function to help with pre_segment_filtering

def filter_with_memory(value,high,low ,a,b, attenuation=100) : #b = (high - low/100) * (high-low) ; mlow = low/100 ; a = low/100 - low * b
    #This function is used to not do the calculation of the parameters everytime
    if value>high : return high*(1-1/attenuation)+value/attenuation
    elif value<low : return value/attenuation
    else : return a + b * value


def pre_segment_filtering(filepath,output_path,higherbound,lowerbound, frames,attenuation=100):
    '''
    Applies a filter on an image before segmenting it, in order to improve the algorithm's precision
    The filter has 3 affine domains :
        -Strong attenuation from 0 to lowerbound, f=x/attenuation
        -Strong attenuation of the excess after higherbound, f=higherbound + (x-higherbound)/attenuation
        -Link between the two

    filepath leads directly to the tiff file of interest
    '''
    img_array = tiff.imread(filepath)
    mlow = lowerbound/attenuation
    b=(higherbound-mlow)/(higherbound-lowerbound)
    a=mlow-lowerbound*b
    filter_array=[filter_with_memory(i, higherbound, lowerbound, a, b, attenuation=attenuation) for i in range(500)]
    T,X,Y=np.shape(img_array)
    img_array=img_array[:min(T,frames)]
    for t in tqdm(range(min(T,frames))):
        for x in range(X):
            for y in range(Y):
                value =  img_array[t,x,y]
                if value<500:
                    img_array[t,x,y] = filter_array[value]
                else:    
                    img_array[t,x,y]=filter_with_memory(img_array[t,x,y], higherbound, lowerbound, a, b, attenuation=attenuation)
    io.imsave(output_path+f"_{higherbound}_{lowerbound}_attenuation{attenuation}.tif",img_array)
    print('Job done :)')