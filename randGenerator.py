import csv
import itertools
import sys
import re
import matplotlib.pyplot as plt
import numpy as np

f = sys.argv[1] # csv file
labels = sys.argv[2] # label file

different = []



def randPlot(f, labels):

    header, correct = 0, 0

    fRead = open(f, 'r')
    reader = csv.reader(fRead)
    row_count = sum(1 for row in fRead) - 1
    fRead.close()


    # sample = 27106
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
            if row[0][-1] == '0':
                correct = int(row[0][0])
            else:
                correct = int(row[0][0]) * 10 + int(row[0][2])
            break
        else:
            i += 1
    print("sample number: " + str(sample))
    print("header: " +str(len(header)))
    print("correct: " + str(correct))
    print("vals: " + str(len(vals)))

    values = [float(v) for v in vals][1:]
    axis = [int(h) for h in header[1:]]
    bars= plt.bar(axis, values)

    max_val = max( (v, i) for i, v in enumerate(values) )[1]


    plt.xticks(np.arange(0, 20))
    bars[correct].set_color('r')
    print("max_val " + str(max_val))
    if not correct == max_val:
        different.append(sample)
    
    plt.title(sample)
    ax = plt.gca()
    ax.set_ylim([0, 1])
    plt.show()

randPlot(f, labels)

def count():
    fRead = open(f, 'r')
    reader = csv.reader(fRead)
    i = 0
    for row in reader:
        i += 1
    print("lines in probs.csv: " + str(i))
    fRead.close()

    fRead = open(labels, 'r')
    reader = csv.reader(fRead)
    i = 0
    for row in reader:
        i += 1
    print("lines in labels: " + str(i))
    fRead.close()

# count()
def findWrong(f, labels):
    i = 0
    fRead = open(f, 'r')
    reader = csv.reader(fRead)
    labels = open(labels, 'r')
    labels2 = csv.reader(labels)
    labels_list = list(labels2)

    for row in reader:
        if i == 0:
            header = row
            i += 1
        else: 
            vals = row
            values = [float(v) for v in vals][1:]
            max_val = max( (v, i) for i, v in enumerate(values) )[1]
            if labels_list[i-1][0][-1] == '0':
                correct = int(labels_list[i-1][0][0])
            else:
                correct = int(labels_list[i-1][0][0]) * 10 + int(labels_list[i-1][0][2])

            if not correct == max_val:
                print("categorized as: " + str(max_val))
                print("correct: " +str(correct))
                different.append(i-1)

            i += 1

    
    print("list of samples: " + str(different))
    print("length: " + str(len(different)))

    fRead.close()
# findWrong(f, labels)

# count()




