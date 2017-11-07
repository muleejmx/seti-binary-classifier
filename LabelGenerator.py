# Imports
import numpy as np
import fitsio
import matplotlib

matplotlib.use('Agg')
import numpy
import pickle
import matplotlib.pyplot as plt

ext = '.fits'
data_loc = './fits_data/'
dataset = [] # name of on pairs
positive_data = [] # pairs share different label
negative_data = [] # pairs share same label

with open('./data/labelled_files.pkl', 'rb') as f:
    label_dict = pickle.load(f) # name:label

for k in label_dict.keys():
    if k[-3:] != 'OFF': # ON
        dataset.append(k)
dict_map = {0:'narrow diagonal (1)', 1:'narrow diagonal (1+)',
            2:'narrow horizontal',
            3:'narrow vertical (1)', 4:'narrow vertical (even)', 5:'narrow vertical (inf)',
            6:'wide diagonal (1)', 7:'wide diagonal (1+)',
            8:'wide horizontal',
            9:'wide vertical (1)', 10:'wide vertical (even)', 11:'wide vertical (inf)',
            12:'no_sig', 13:'comb_sig', 14:'signal of interest'}

print("num of pairs: " + str(len(dataset)))

same_counter, diff_counter = 0, 0

for on in dataset:
    
    off = on + '_OFF'

    on_data = fitsio.read(data_loc + str(on) + ext)
    off_data = fitsio.read(data_loc + str(off) + ext)

    plt.subplot(211)
    plt.suptitle(on,fontsize='8')
    plt.title('label= '+str(label_dict[on])+' '+dict_map[label_dict[on]],fontsize='8')
    plt.imshow(on_data)
    plt.subplot(212)
    plt.title('label= '+str(label_dict[off])+' '+dict_map[label_dict[off]],fontsize='8')
    plt.imshow(off_data)
    
    plt.subplots_adjust(bottom=0.05, right=0.8, top=0.9)

    cax = plt.axes([0.85, 0.1, 0.075, 0.8])
    plt.colorbar(cax = cax)

    if str(label_dict[on]) == str(label_dict[off]):
    	plt.savefig('./label_examples/same_label/temp' + str(same_counter))
    	same_counter += 1
    else:
    	plt.savefig('./label_examples/different_label/temp' + str(diff_counter))
    	diff_counter += 1
    plt.close()

