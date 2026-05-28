import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tifffile
import time
import pandas as pd


## INPUT USER --------------------------------
images_path = r"/Users/laureleblanc/Desktop/Collective-twitching-2026/5_competition_assay/data_preprocessing/3_Rotated_for_use"
scale_path = r"/Users/laureleblanc/Desktop/Collective-twitching-2026/5_competition_assay/data_preprocessing/1_Raw_images"
csv_path=r"/Users/laureleblanc/Desktop/Collective-twitching-2026/5_competition_assay/data_preprocessing/Output"
# --------------------------------------------


#Global definitions
sampling_size=14 #step length with which we segment the leasding edge. 14 is close to bacterial size
leading_direction = np.array([-1, 0])
bact_size=3 #bacterial width in pixels @20X
colors={'260427':'darkblue'} 
strands={'2619':'WT', '2621':'ΔpilH', '2624':'ΔpilG'}
colors1=['red', 'green']
order={('2619+2621', '2h30'): '3_', ('2619+2624', '2h30'):'4_' ,('2619+2621', '10min'): '1_', ('2619+2624', '10min'): '2_'}
supercolor=[['lightcoral','brown','coral','sandybrown'],['mediumspringgreen','mediumseagreen','olivedrab','yellowgreen']]

# Functions

def detect_first_white(data, position, leading_direction,sampling_size):
    """
    Gives the first abscissa of a non-black pixel, in the frame of reference centered
    on the 'position' vector with the two generators y=leading direction;
    x = orthogonal vector
    Leading direction follows the edge in the trigonometric direction. 
    Function intended to use on images where the leading edge is on the right of the image and the heart of the colony on the left.
    """
    ortho = np.array([-leading_direction[1], leading_direction[0]])
    x=0
    point = position
    while (0<=point[0]<data.shape[0]) and (0<=point[1]<data.shape[1]) :
        for k in range(sampling_size) :
            if data[tuple(point + k * leading_direction)]>0 :
                return point[1]
        point = point + ortho
        x+=1
    return x

def find_leading_edge_right(data, sampling_size, leading_direction=np.array([-1, 0])):
    """
    Parameters
    ----------
    data : np array description of the image
    
    sampling_size : int
        Step for segmenting of leading edge
    leading_direction : np array
        Vector, needs to be pointing north for the intended images

    Returns
    -------
    abs_list : Python list describing the leading edge. Reads from south to north of image. 
               Value is the distance to the right-edge of the image.

    """
    position=np.array([data.shape[0]-1, data.shape[1]-1]) #Starting from south east corner
    abs_list=[]
    count=0
    while (0<=position[0]<data.shape[0]) and (0<=position[1]<data.shape[1]) :
        abs_list.append((count, detect_first_white(data, position, leading_direction, sampling_size)))
        position+= sampling_size * leading_direction
        count+=1
    return abs_list

def edge_check(data,leading_direction, sampling_size, thickness = 2):
    """
    Plots the image with the edge-finding done, if you want to verify it is working well. Not useful in the main pipeline
    """
    plt.imshow(data, cmap='gray', origin='upper')
    for i, xprime in enumerate(find_leading_edge_right(data, sampling_size, leading_direction)):
        _,x = xprime
        y_start = data.shape[0]-1 -i * sampling_size
        y_end = max(y_start-sampling_size+1, 0)
        if 0 <= x < data.shape[1]:  # éviter de sortir de l'image
            plt.plot([x, x], [y_start, y_end], color='red', linewidth=thickness)
    plt.show()

# edge_check(img_array, leading_direction, sampling_size)

def find_scale_micron(file_name):
    """
    Finds the scale of the tif files we are working with. Since we rotated them, they lost it in their metadata
    so we need to extract it from the source tif files.
    Xresolution (assuming Xresolution = Yresolution) : tuple
        We can fit XResolution[0] pixels in Xresolution[1] microns.
    """
    with tifffile.TiffFile(file_name) as tif:
        tif_tags = {}
        for tag in tif.pages[0].tags.values():
            name, value = tag.name, tag.value
            tif_tags[name] = value
        return tif_tags['XResolution']


def averaged_graphs(folder_path, sampling_size=14, depth=1600, margin=50):
    """
    Extracts all tif files in given folder, finds the leading edge and creates the intensity average accross leading edge
    This is the main time-consuming function

    folder_path : directory path of a folder with two files per image (1 per channel)
        The red channel files are labeled 'C2_YYMMDD_2619_262X....'
        The green channel files are labeled 'C3_YYMMDD_2619_262X....'
    """
    tif_files = [f for f in os.listdir(folder_path) if f.endswith('.tif')]
    datas = []
    for file in tif_files : 
        img = Image.open(os.path.join(folder_path ,file))
        img_array = np.array(img)
        datas.append((file, img_array))
    c2dict,c3dict = {},{}
    results_green, results_red = {},{}
    for (name, data) in datas:  #filtering between red and green, and adapting label so that we can analyze two channels compared to the same leading edge
        if name.startswith('C2-'): #'C2-'
            c2dict[name[3:]]=data
        elif name.startswith('C1-'): #'C3-'
            c3dict[name[3:]]=data
    for label in c2dict :
        xs_c2 = find_leading_edge_right(c2dict[label], sampling_size)
        xs_c3 = find_leading_edge_right(c3dict[label], sampling_size)
        xs = [max(xs_c2[i], xs_c3[i]) for i in range(len(xs_c2))]
        #initial values, to which we'll add bacterial presence :
        results2=np.zeros(depth+margin+1)
        results3=np.zeros(depth+margin+1)
        for i,x in xs:
            #correction if the depth goes out of bounds
            complement_left = [0 for _ in range(-x+depth)]
            complement_right = [0 for _ in range (x+margin+1-c2dict[label].shape[1])]
            for k in range(1,sampling_size+1):
                results2+=np.array(complement_right + [c2dict[label][c2dict[label].shape[0]-(i*sampling_size+k),x+margin-dx]!=0 for dx in range(margin+depth+1) if 0<=x+margin-dx<c2dict[label].shape[1]]+complement_left) #Make Xs a list of points, not of absissas !!
                results3+=np.array(complement_right + [c3dict[label][c3dict[label].shape[0]-(i*sampling_size+k),x+margin-dx]!=0 for dx in range(margin+depth+1) if 0<=x+margin-dx<c3dict[label].shape[1]]+complement_left)
        results_red[label]=results2/bact_size
        results_green[label]=results3/bact_size
    return (results_red, results_green)

def data_treatment(folder_path, averaged = None, sampling_size=14, depth = 1600, margin = 50, window_sizes=[1]):
    """
    This function does the first data analysis :
        It averages the data across several pixel to reduce variability
        It finds and adapts the scale of each image
    window_sizes is an array with all the different lengths with which we want to locally average. 
        1 corresponds to an unfiltered curve
        10 is around bacterial size with 20X focal
    By default, returns all the curves (without filtering).
    """
    if averaged == None :
       dred, dgreen = averaged_graphs(folder_path, sampling_size, depth, margin)
    else: dred, dgreen = averaged
    #for label in blacklist:
        #if label in dred.keys():
            #del dred [label]
            #del dgreen [label]
    results={}
    for i, label in enumerate(dred):
        scale=find_scale_micron(os.path.join(scale_path ,label))
        absissa=np.array([scale[1]/scale[0]*(x-margin) for x in range(depth+margin+1)])  #Might be a +/-1 mistake in the indexs...
        for j, k in enumerate(window_sizes):
            convole_red=np.convolve(dred[label], np.ones(k)/k, mode='same')
            extract_red=convole_red[[mu*k for mu in range(len(convole_red)//k)]]
            convole_green=np.convolve(dgreen[label], np.ones(k)/k, mode='same')
            extract_green=convole_green[[mu*k for mu in range(len(convole_green)//k)]]
            adapted_absissa=absissa[[mu*k for mu in range(len(convole_red)//k)]]
            results[label]=(adapted_absissa, extract_red, extract_green)
    return results
        
def create_analysis_groups(folder_path):
    """
    Makes a dictionnary that groups together the similar conditions.
    This helps with labeling the outputs of the main function
    Keys : (sample_type ; time post incubation)
    Needs filenames to follow the same rules :
        date_sampletype_XXX_mixNNpc_timepostincub_XXX.tif
    """
    tif_files = [f for f in os.listdir(folder_path) if f.endswith('.tif')]
    datas = []
    for file in tif_files : 
        img = Image.open(os.path.join(folder_path ,file))
        img_array = np.array(img)
        datas.append((file, img_array))
    groups={}
    for (name, data) in datas:  #filtering between red and green, and adapting label so that we can analyze two channels compared to the same leading edge
        if name.startswith('C2-'):
            label=name[3:]
            separation1 = label.find('_')
            separation2 = label.find('_',separation1+1)
            time_index = label.find('pc_', label.find('mix'))+3
            sample_type = label[separation1+1:separation2]
            timepoint=label[time_index:label.find('_', time_index)]
            key=(sample_type,timepoint)
            if key in groups.keys() :
                groups[key].append(label)
            else:
                groups[key]=[label]
    return groups




def results_printing(folder_path, window_size=10, absissa_max = 300, plotting=False, print_to_csv=False, normalize_mean=False):
    results=data_treatment(folder_path, averaged=averaged, window_sizes=[window_size])
    for key in groups :
        labels=groups[key] #python list of relevant labels
        scale = results[labels[0]][0]
        thresh = next((i for i, x in enumerate(scale) if x > absissa_max), -1) # X threshold to limit ourselves to the depth that we want
        scale = scale[:thresh]
        curves1, curves2 =[(results[label][1])[:thresh] for label in labels],[(results[label][2])[:thresh] for label in labels]
        curves = np.array((curves1, curves2))
        if normalize_mean : 
            for i in range(len(curves1)):
                normalizer = np.mean(curves1[i]+curves2[i])
                curves[:,i]/=normalizer
        #now making the comparison curves, where y = intensity/red+green
        normalizer = np.array([[curves[0,i,k]+curves[1,i,k] for k in range(len(curves[0][0]))] for i in range(len(curves[0]))])
        for i in range(len(normalizer)):
            for j,value in enumerate(normalizer[i]):     #this part is to avoid dividing by zero when normalizing in places where both channel values are zero
                if value==0 : normalizer[i,j]=1
        curves_compared = np.array(([[curves[0,i,j]/normalizer[i,j] for j in range(len(normalizer[0]))] for i in range(len(normalizer))], [[curves[1,i,j]/normalizer[i,j] for j in range(len(normalizer[0]))] for i in range(len(normalizer))]))
        mean =np.mean(curves, axis=1)          #averaging across labels     
        std=np.std(curves, axis=1)
        mean_compared = np.mean(curves_compared, axis=1)
        std_compared = np.std(curves_compared, axis=1)
        if print_to_csv: 
            indices = [(0,j) for j in range(len(curves1))] + [(1,j) for j in range(len(curves1))]
        #individual curves
            data1 = {'scale' : np.array(scale)}
            for i,j in indices:
                data1[colors1[i]+'_'+labels[j]] = curves[i,j]
            df = pd.DataFrame(data1)
            #print(df.columns)
            df.to_csv(os.path.join(r"/Users/laureleblanc/Desktop/PersatLAB/5_Analysis/Code/17_Competition_Pa/data_processing/Output/csv/1_Individual_curves",order[key]+key[0] + '_'+key[1]+"_individual.csv"))     
        #mean curves
            data2 = {'scale' : np.array(scale)}
            for i in range(2):
                data2[colors1[i]+'_mean']=mean[i]
                data2[colors1[i]+'_std']=std[i]
            df = pd.DataFrame(data2)
            df.to_csv(os.path.join(r"/Users/laureleblanc/Desktop/PersatLAB/5_Analysis/Code/17_Competition_Pa/data_processing/Output/csv/2_Mean",order[key]+key[0] + '_'+key[1]+"_normalised.csv"))     
        #compared curves
            data3 = {'scale' : np.array(scale)}
            for i in range(2):
                data3[colors1[i]+'_mean']=mean_compared[i]
                data3[colors1[i]+'_std']=std_compared[i]
            df = pd.DataFrame(data3)
            df.to_csv(os.path.join(r"/Users/laureleblanc/Desktop/PersatLAB/5_Analysis/Code/17_Competition_Pa/data_processing/Output/csv/3_Compared",order[key]+key[0] + '_'+key[1]+"_compared.csv"))     
            print('Job done : csv '+key[0] + '_'+key[1])
        if plotting:
            fig = plt.figure(figsize=(30,20))
            gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1])  # 2 lignes, 2 colonnes
            ax1 = fig.add_subplot(gs[0, :])
            ax2 = fig.add_subplot(gs[1, :])
            ax3 = fig.add_subplot(gs[2, 0])
            ax4 = fig.add_subplot(gs[2, 1])
            axs=[ax3,ax4]
            handles,names=[],[]
            handles_compared,names_compared=[],[]
            for i  in range(2):
                handle, = ax1.plot(scale, mean[i], color=colors1[i], linewidth=6)
                handle_compared, = ax2.plot(scale, mean_compared[i], color=colors1[i], linewidth=6)
                    # If plotting the std :
                # ax1.fill_between(scale, mean[i]-std[i], mean[i]+std[i], alpha=0.2)
                # ax2.fill_between(scale, mean_compared[i]-std_compared[i], mean_compared[i]+std_compared[i], alpha=0.2)
                    # If plotting individual curves :
                for k in range(len(curves[i])):
                    ax1.plot(scale, curves[i][k], color=supercolor[i][k], linewidth=1, alpha=0.8)
                    ax2.plot(scale, curves_compared[i][k], color=supercolor[i][k], linewidth=1, alpha=0.8)
                ax1.set_title(key[0] + ' t=' + key[1] + f' Number of replicates = {len(curves1)}')
                handles.append(handle)
                names.append(strands[key[0][5*i:5*i+4]])
                handles_compared.append(handle_compared)
                names_compared.append(strands[key[0][5*i:5*i+4]])
            ax1.legend(handles, names, bbox_to_anchor=(1.08, 0.5))
            ax1.invert_xaxis()
            ax2.legend(handles_compared, names_compared, bbox_to_anchor=(1.08, 0.5))
            ax2.invert_xaxis()
            if normalize_mean : ax1.annotate(r'$\mathbf{Normalized \,\, bacterial \, \, count \,\, :}$'+'\n' + r'$\mathbf{\frac{Count}{Mean(Red+Green)}}$', xy=(-0.15,0.5), xycoords='axes fraction', annotation_clip=False)
            else : ax1.annotate(r'$\mathbf{Bacterial \,\, count}$', xy=(-0.15,0.5), xycoords='axes fraction', annotation_clip=False)
            ax2.annotate(r'$\frac{\mathbf{Count \,\, of \,\, Bacteria}}{\mathbf{Red \, count + Green \, count}}$', xy=(-0.15,0.5), xycoords='axes fraction', annotation_clip=False)
            for i  in range(2):
                handles,names=[],[]
                axs[i].set_title(f'{strands[key[0][5*i:5*i+4]]}')
                for j,label in enumerate(labels):
                    handle, = axs[i].plot(scale, curves[i][j], color=colors[label[:6]])
                    handles.append(handle)
                    names.append(label)
                axs[i].invert_xaxis()
                axs[i].legend(handles, names, loc='lower left', fontsize=6)
            plt.tight_layout()
            plt.savefig(os.path.join(csv_path, "/Plots/combined", order[key] + key[0] + '_'+key[1]+'.png'))
            plt.close()
            print('Job done : plotting '+key[0] + '_'+key[1])
    print('Job done')



groups=create_analysis_groups(images_path)


#We create 'averaged' now because it is the most computation-heavy step, and we want to experiment with the plots later without re-doing it 
averaged = averaged_graphs(images_path)


results_printing(images_path, absissa_max=350, plotting=True, print_to_csv = True, normalize_mean=True)
    