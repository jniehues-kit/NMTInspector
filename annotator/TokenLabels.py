#!/usr/bin/env python


import sys


class TokenLabels:

    def __init__(self, data,filename):
        self.data = data
        self.filename = filename

    def annotate(self):
        f = open(self.filename)
        line = f.readline().strip()
        for i in range(len(self.data.sentences)):
            if(not line):
                print("Not enough labels for the data set")
                exit(-1);
            labels = line.strip().split()
            if(len(labels) != len(self.data.sentences[i].words)):
                print ("Not enough labels in line")
                print("Sentence:", i, "Length of tokens: ",len(labels)," Length hidden: ",len(self.data.sentences[i].words))
                exit(-1)
            self.data.sentences[i].labels = labels
            line = f.readline().strip()


def annotate(data,filename):
    a = TokenLabels(data,filename)
    a.annotate()
