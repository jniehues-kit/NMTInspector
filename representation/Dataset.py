#!/usr/bin/env python


import sys

import numpy

class Dataset:
    def __init__(self,label):
        self.label = label
        self.sentences = []

class Sentence:
    def __init__(self,rep):
        self.data = rep;
        self.label = ""
        #self.word_label = [""] * self.data.size[0]

    def addWord(self,rep,l):
        newSize = (self.data.shape[0]+1,self.data.shape[1])
        self.data = numpy.resize(self.data,newSize)
        self.data[-1] = rep
        self.words.append(l)
