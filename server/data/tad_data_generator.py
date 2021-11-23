#%%

import numpy as np
import csv
from iced import normalization, filter
import imageio
import os
import cooler

#%%

def cool2mat(cool_filename: 'str', chr: 'str', resolution: 'int'):
    '''
    '''
    c = cooler.Cooler(f'./{cool_filename}.mcool::resolutions/{resolution}')
    mat = c.matrix(balance=False).fetch(chr)
    np.savetxt(f'./{cool_filename}.matrix', mat)

cool2mat('U54-HFFc6-FA-DSG-MNase-R1-R3.hg38.mapq_30.500', 'chr7', '10000')

#%%
#%%
img_size = 64
matrix_file = 'HFFc6_chr7_10k.matrix'
# TAD file is generated through OnTAD algorithm
tad_file = 'OnTAD_HFFc6_pen0.1_max200_10k_chr7.tad'
dir_name = 'TAD_HFFc6_chr7_10k'

if not os.path.exists(dir_name):
    os.makedirs(dir_name)


matrix = np.loadtxt(matrix_file)
matrix = filter.filter_low_counts(matrix, percentage=0.04)
matrix = normalization.ICE_normalization(matrix)
matrix[ matrix == 0 ] = 1 # prevent zero cells when log 

max_logv = np.log10(np.amax(matrix))

#%%
f = open(os.path.join(dir_name, 'label.csv'), 'w')
label_writer = csv.writer(f)
header = ['img', 'start', 'end', 'level', 'mean', 'score']
label_writer.writerow(header)

tad_f = open(tad_file)
row_reader = csv.reader(tad_f, delimiter='\t')
for idx, row in enumerate(row_reader):
    # [start, end, tad_level, mean, score] = row
    [start, end, tad_level] = [int(i) for i in row[:3]]
    [mean, score] = [float(i) for i in row[3:]]

    if tad_level == 0:
        continue
    
    TAD = matrix[start:end, start:end]
    # log scale TADs to [0,1]
    TAD = np.log10(TAD)/max_logv
    imageio.imwrite(os.path.join(dir_name, f'{idx}.jpg') , TAD)
    
    label_writer.writerow([idx, start, end, tad_level, mean, score])

f.close()
tad_f.close()
# %%



