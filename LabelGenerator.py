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

print("num of pairs: " + str(len(dataset)))

same_counter, diff_counter = 0, 0

for on in dataset:
    
    off = on + '_OFF'

    on_data = fitsio.read(data_loc + str(on) + ext)
    off_data = fitsio.read(data_loc + str(off) + ext)

    plt.subplot(211)
    plt.suptitle(on)
    plt.title(str(label_dict[on]))
    plt.imshow(on_data)
    plt.subplot(212)
    plt.title(str(label_dict[off]))
    plt.imshow(off_data)
    
    plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)

    cax = plt.axes([0.85, 0.1, 0.075, 0.8])
    plt.colorbar(cax = cax)

    if str(label_dict[on]) == str(label_dict[off]):
    	plt.savefig('./label_examples/same_label/temp' + str(same_counter))
    	same_counter += 1
    else:
    	plt.savefig('./label_examples/different_label/temp' + str(diff_counter))
    	diff_counter += 1
    plt.close()

