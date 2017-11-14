import csv
import itertools
import sys
import re
import matplotlib.pyplot as plt
import numpy as np

orig = sys.argv[1] # terminal file
txt = sys.argv[2] # stripped txt file
f = sys.argv[3] # csv file
labels = sys.argv[4] # label file

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
            '-11', '-10', '-9', '-8', '-7', '-6', '-5', '-4', '-3', '-2', '-1',
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11') )
        i = 0
        row = [str(i)]
        for line in txt:
            words = line.replace("[", "").replace("]", "")
            words = words.split()

            for word in words:
                if len(row) < 24:
                    row.append(word)
                else:
                    writer.writerow(row)
                    i += 1
                    row = [str(i)]
    finally:
        f.close()

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
            if row[0][0] != '-':
                correct = int(row[0][0])
            else:
                correct = int(row[0][:2])
            break
        else:
            i += 1
    print("row count: " + str(row_count))
    print("sample number: " + str(sample))
    print("header: " +str(header))
    print("correct: " + str(correct))

    bars= plt.bar([int(h) for h in header[1:]], [float(v) for v in vals][1:])
    bars[correct+11].set_color('r')
    plt.xticks(np.arange(-11, 12))
    plt.title(sample)
    ax = plt.gca()
    ax.set_ylim([0, 1])
    plt.show()

strip(orig, txt)
writeCSV(txt, f)
randPlot(f, labels)


