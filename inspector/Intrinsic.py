#!/usr/bin/env python


import sys
from numpy import linalg as LA

class Intrinsic:

    def __init__(self,data):
        self.data = data

    def inspect(self):
        self.calcMeanAndVariation();
        return []

    def calcMeanAndVariation(self):
        mean = {}
        count = {}
        for i in range(len(self.data.sentences)):
            for j in range(len(self.data.sentences[i].words)):
                l = self.data.sentences[i].labels[j]
                h = self.data.sentences[i].data[j]
                if(l in mean):
                    mean[l] += h
                    count[l] += 1
                else:
                    mean[l] = 1
                    count[l] = 1
        for k in mean.keys():
            mean[k] = mean[k]/count[k]
        for k in mean.keys():
            for l in mean.keys():
                if(l != k):
                    diff = LA.norm(mean[k] - mean[l])
                    print ("Mean different of labels",l,"and",k,":",diff)
        diff = {}
        for i in range(len(self.data.sentences)):
            for j in range(len(self.data.sentences[i].words)):
                l = self.data.sentences[i].labels[j]
                h = self.data.sentences[i].data[j]
                d = LA.norm(h - mean[l])
                if(l in diff):
                    diff[l] += d
                else:
                    diff[l] = d
        
        for k in diff.keys():
            diff[k] = diff[k]/count[k]
            print("Variance in cluster",k,":",diff[k])
    
def inspect(data):
    a = Intrinsic(data)
    return a.inspect()
