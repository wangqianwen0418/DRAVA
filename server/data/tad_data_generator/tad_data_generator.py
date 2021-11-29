#%%

import numpy as np
import csv
from iced import normalization, filter
import imageio
import os
import cooler
import subprocess

#%%

def cool2mat(cool_filename: 'str', chrs: 'list', resolution: 'int'):
    '''
    '''
    c = cooler.Cooler(f'./{cool_filename}.mcool::resolutions/{resolution}')
    for chr in chrs:
        mat = c.matrix(balance=False).fetch(chr)
        # files are saved at scratch3
        np.savetxt(f'/n/scratch3/users/q/qiw433/tad_data/{cool_filename}_{chr}_{int(resolution/1000)}k.matrix', mat)

#%%
# generate tads from the matrix files

def generateTADs():
    bashCommand = f'''
    for i in {{1..10}}
    do 
        ./OnTAD /n/scratch3/users/q/qiw433/tad_data/{cool_filename}_chr${{i}}_10k.matrix -penalty 0.1 -maxsz 400 -minsz 40 -o /n/scratch3/users/q/qiw433/tad_data/OnTAD_{cool_filename}_chr${{i}}_10k
    done
    '''
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
#%%

def saveDataset(matrix_file, tad_file, dir_name, chr):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    matrix = np.loadtxt(matrix_file)
    matrix = filter.filter_low_counts(matrix, percentage=0.04)
    matrix = normalization.ICE_normalization(matrix)
    matrix[ matrix == 0 ] = 1 # prevent zero cells when log 

    max_logv = np.log10(np.amax(matrix))

    #%%
    f = open(os.path.join(dir_name, 'label.csv'), 'a+')
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
        imageio.imwrite(os.path.join(dir_name, f'{chr}:{idx}.jpg') , TAD)
        
        label_writer.writerow([f'{chr}:{idx}', start, end, tad_level, mean, score])

    f.close()
    tad_f.close()
# %%

if __name__ == "__main__": 
    chrs = [f'chr{i}' for i in range(1, 10) ]
    cool_filename = 'U54-HFFc6-FA-DSG-MNase-R1-R3.hg38.mapq_30.500'
    resolution = 10000
    dataset_dir_name = '../TAD_HFFc6_10k'

    #%%
    # cool2mat(cool_filename, chrs, resolution)
    # generateTADs()



    #%%

    # convert tads to image datasets

    for chr in chrs :
        matrix_file = f'/n/scratch3/users/q/qiw433/tad_data/{cool_filename}_{chr}_{int(resolution/1000)}k.matrix'
        tad_file = f'/n/scratch3/users/q/qiw433/tad_data/OnTAD_{cool_filename}_{chr}_{int(resolution/1000)}k.tad'
        saveDataset(matrix_file, tad_file, dataset_dir_name, chr)

