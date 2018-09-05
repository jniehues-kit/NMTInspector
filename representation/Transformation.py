#!/usr/bin/env python


import numpy as np

def transformAlignmentMatrix(data):

    maxSize=250

    for i in range(len(data.sentences)):
        s=data.sentences[i].data.shape
        data.sentences[i].data = np.pad(data.sentences[i].data,((0,maxSize-s[0]),(0,maxSize-s[1])),"constant").reshape(1,-1)
        data.sentences[i].words = ["matrix"]