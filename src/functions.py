import numpy as np
from collections import defaultdict
# import matplotlib.pyplot as plt
import random
import datetime
from time import time


def select_lines(file_in, file_out, db_size, shuffled=False):
    with open(file_in, "r") as input_file:
        scanned = input_file.readlines()
        if shuffled:
            random.shuffle(scanned)
    with open(file_out, "w") as output_file:
        for line in scanned[:db_size]:
            output_file.write(line)


def print_matrix(matrix, name):
    print "Matrix ", name, ":"
    for i in range(np.size(matrix, 0)):
        print matrix[i]


def serialize_matrix(matrix, mfile):
    with open(mfile, "w") as mlog:
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                mlog.writelines(str(matrix[i][j]) + " ")
            mlog.writelines("\n")


def filter_dataset(source, target, min_len, is_random=False):

    word_freq = defaultdict(int)
    words = 0
    output_lines = 0
    with open(source, 'r') as input_file:
        scanned = input_file.readlines()
    with open(target, 'w') as output_file:
        for line in scanned:
            record = line.rstrip().split(' ')
            if len(record) < min_len:
                continue
            output_file.write(line)
            for elem in record:
                i = int(elem)
                word_freq[i] += 1
            words += len(record)
            output_lines += 1
    if is_random:
        with open(target, 'r') as f:
            scanned = f.readlines()
        random.shuffle(scanned)
        with open(target, 'w') as f:
            f.writelines(scanned)

    avg_doc_len = float(words) / float(output_lines)

    d = sorted(word_freq.iteritems(), key=lambda (k, v): v, reverse=True)[:40]
    for item in d:
        print item
    word = []
    frequency = []
    for i in range(len(d)):
        word.append(d[i][0])
        frequency.append(d[i][1])
    indices = np.arange(len(d))
    # plt.bar(indices, frequency, color='r')
    # plt.xticks(indices, word, rotation='vertical')
    # plt.tight_layout()
    dt = datetime.datetime.fromtimestamp(int(time()))
    datee = str(dt.year) + str(dt.month) + str(dt.day) + str(dt.hour) + str(dt.minute)
    figname = "DATABASES/PLOTS/" + target[10:-4] + "_" + datee + ".png"
    # plt.savefig(figname)

    with open('DATABASES/dblist.txt', 'a') as logfile:
        logfile.write('%s\t%s\t%2d\t%6d\t%2.1f\t%s\n'
                      % (target, source, min_len, output_lines, avg_doc_len, is_random))

    with open('DATABASES/dblist.csv', 'a') as logfile:
        s = '%s;%s;%d;%d;%s;%s\n' \
            % (target, source, min_len, output_lines, str(avg_doc_len).replace('.', ','), is_random)
        # s = s.replace('.', ',')
        logfile.write(s)


def analyse_dataset(dataset):

    record_freq = defaultdict(int)
    with open(dataset, 'r') as input_file:
        scanned = input_file.readlines()
    for line in scanned:
        record = line.rstrip().split(' ')
        if(len(record)) > 60:
            record_freq[60] += 1
        else:
            record_freq[len(record)] += 1

    d = sorted(record_freq.iteritems(), key=lambda (k, v): k)
    for item in d:
        print item
    word = []
    frequency = []
    for i in range(len(d)):
        word.append(d[i][0])
        frequency.append(d[i][1])
    indices = np.arange(len(d))
    # plt.bar(indices, frequency, color='r')
    # plt.xticks(indices, word, rotation='vertical')
    # plt.tight_layout()
    dt = datetime.datetime.fromtimestamp(int(time()))
    datee = str(dt.year) + str(dt.month) + str(dt.day) + str(dt.hour) + str(dt.minute)
    figname = dataset[14:-4] + "_histogram_" + datee + ".png"
    # plt.savefig(figname)

