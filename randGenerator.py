import csv
import itertools
import sys
import re
import matplotlib.pyplot as plt
import numpy as np

f = sys.argv[1] # csv file
labels = sys.argv[2] # label file


def randPlot(f, labels):
    header, correct = 0, 0

    fRead = open(f, 'r')
    reader = csv.reader(fRead)
    row_count = sum(1 for row in fRead) - 1
    fRead.close()

    sample = np.random.randint(0, row_count)
    i = 0

    fRead = open(f, 'r')
    reader = csv.reader(fRead)
    for row in reader:
        if i == 0:
            header = row
            i += 1
        elif i == sample + 1:
            vals = row
            break
        else:
            i += 1
    fRead.close()

    labels = open(labels, 'r')
    reader = csv.reader(labels)
    i = 0
    for row in reader:
        if i == sample:
            print(row)
            if row[0][-1] == '0':
                correct = int(row[0][0])
            else:
                correct = int(row[0][0]) * 10 + int(row[0][2])
            break
        else:
            i += 1
    print("row count: " + str(row_count))
    print("sample number: " + str(sample))
    print("header: " +str(len(header)))
    print("correct: " + str(correct))
    print("vals: " + str(len(vals)))


    bars= plt.bar([int(h) for h in header[1:]], [float(v) for v in vals][1:])
    plt.xticks(np.arange(0, 20))
    bars[correct].set_color('r')
    
    plt.title(sample)
    ax = plt.gca()
    ax.set_ylim([0, 1])
    plt.show()

randPlot(f, labels)


