import csv
import itertools
import sys
import re
import matplotlib.pyplot as plt
import numpy as np


orig = sys.argv[1] # terminal file
txt = sys.argv[2] # stripped txt file
f = sys.argv[3] # csv file
# labels = sys.argv[4] # label file
# names = sys.argv[5] # names file

different = []


def strip(orig, txt):
    write = False
    excess = "INFO:tensorflow:probabilities = "
    exclude = ["INFO:tensorflow:Using", "mulan", "of", "computations", "evaluation",
    "{", "}", "INFO:tensorflow:Finished", "INFO:tensorflow:Saving", "device"]

    orig = file(orig)
    txt = open(txt, 'w+')

    for line in orig:

        if write:
            for e in exclude:
                if e in line:
                    write = False
            if write:
                line = line.replace(excess, "")
                line = re.sub('\(.*?\)','', line)
                txt.write(line)
            if ']' in line:
                write = False
        elif '[' in line:
            write = True
            for e in exclude:
                if e in line:
                    write = False
            if write:
                line = line.replace(excess, "")
                line = re.sub('\(.*?\)','', line)
                txt.write(line)

    txt.close()

def writeCSV(txt, f):
    txt = file(txt)
    f = open(f, 'w+')
    samp = 0

    try:
        writer = csv.writer(f)
        writer.writerow(('Sample',
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
            '11', '12', '13', '14', '15', '16', '17', '18', '19') )
        i = 0
        row = [str(i)]
        for line in txt:
            words = line.replace("[", "").replace("]", "")
            words = words.split()

            for word in words:
                if len(row) < 21:
                    row.append(word)
                else:
                    writer.writerow(row)
                    i += 1
                    row = [str(i), word]
    finally:
        f.close()

def randPlot(f, labels):
    header, correct = 0, 0

    # sample = 25503
    sample = np.random.randint(0, row_count)
    i = 0

    names = open(names, 'r')
    names2 = csv.reader(names)
    names_list = list(names2)

    fRead = open(f, 'r')
    reader1 = csv.reader(fRead)
    fRead_list = list(reader1)
    vals = fRead_list[sample+1]
    # for row in reader:
    #     if i == 0:
    #         header = row
    #         i += 1
    #     elif i == sample + 1:
    #         vals = row
    #         break
    #     else:
    #         i += 1
    # fRead.close()

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



strip(orig, txt)
writeCSV(txt, f)
# findWrong(f, labels)
# randPlot(f, labels)



