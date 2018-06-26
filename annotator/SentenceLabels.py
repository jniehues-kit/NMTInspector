#!/usr/bin/env python


import sys


class SentenceLabels:

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
            labels = []
            for j in range(len(self.data.sentences[i].words)):
                labels.append(line)
            self.data.sentences[i].labels = labels
            self.data.sentences[i].label = line
            line = f.readline().strip()
            # print ("Hidden:",self.data.sentences[i].data.shape)
            # print ("Sentences:", " ".join(self.data.sentences[i].words))
            # print ("Labels:", " ".join(self.data.sentences[i].labels))


def annotate(data,filename):
    a = SentenceLabels(data,filename)
    a.annotate()
