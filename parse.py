import csv
import itertools
import sys
import re
import matplotlib.pyplot as plt
import numpy as np

orig = sys.argv[1] # terminal file
txt = sys.argv[2] # stripped txt file
f = sys.argv[3] # csv file

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
                    row = [str(i)]
    finally:
        f.close()

strip(orig, txt)
writeCSV(txt, f)


