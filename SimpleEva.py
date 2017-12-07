#!/usr/bin/python
#coding=utf-8

'''

'''

import sys
import numpy as np
from collections import defaultdict

def read_seq(line):

    seq = [int(item.strip()) for item in line[1:-1].strip().split(' ') if item.strip() != '' and int(item.strip()) != 0]
    #print seq

    return seq

t,f = defaultdict(int),defaultdict(int)

samples = open(sys.argv[1],'r').read().split('\n\n')
for sample in samples:
    lines = sample.strip().split('\n')
    if len(lines) != 3: continue
    
    seq1 = read_seq(lines[0])
    seq2 = read_seq(lines[1])
    if np.array_equal(np.array(seq1), np.array(seq2)): t[len(seq1)] += 1
    else: f[len(seq2)] += 1

for k in f.keys():
    print k, t[k],f[k]

