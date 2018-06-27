#!/usr/bin/env python

from __future__ import print_function
import sys
from numpy import linalg as LA
from numpy import dot
from numpy import log
from numpy import copy
from numpy import zeros

class Intrinsic:

    def __init__(self,data):
        self.data = data

    def inspect(self):
        self.calcMeanAndVariationWordVectors();
        self.calcMeanActivation();
        self.calcEntropyActivation();
        self.calcHistogram();
        return []

    def calcEntropyActivation(self):
        entropy = {}
        count = {}
        for i in range(len(self.data.sentences)):
            for j in range(len(self.data.sentences[i].words)):
                try:
                    l = self.data.sentences[i].labels[j]
                except AttributeError:
                    l = "none"
                h = self.data.sentences[i].data[j]
                if(l in entropy):
                    entropy[l] += dot(abs(h),log(abs(h)))
                    count[l] += h.size
                else:
                    entropy[l] = dot(abs(h),log(abs(h)))
                    count[l] = h.size
        allEntropy = 0
        allCount = 0
        for l in entropy.keys():
            allCount += count[l]
            allEntropy += entropy[l]
            print ("Entropy of activation of labels",l,":",entropy[l]/count[l])
        print ("Entropy of activation of labels:",allEntropy/allCount)



    def calcMeanActivation(self):
        absmean = {}
        qmean = {}
        count = {}
        for i in range(len(self.data.sentences)):
            for j in range(len(self.data.sentences[i].words)):
                try:
                    l = self.data.sentences[i].labels[j]
                except AttributeError:
                    l = "none"
                h = self.data.sentences[i].data[j]
                if(l in absmean):
                    absmean[l] += sum(abs(h))
                    qmean[l] += dot(h,h)
                    count[l] += h.size
                else:
                    absmean[l] = sum(abs(h))
                    qmean[l] = dot(h,h)
                    count[l] = h.size
        allAbsmean = 0
        allQmean = 0
        allCount = 0
        for l in absmean.keys():
            allAbsmean += absmean[l]
            allQmean += qmean[l]
            allCount += count[l]
            print ("Absolut mean activation of labels",l,":",absmean[l]/count[l])
            print ("Quadratic mean activation of labels",l,":",qmean[l]/count[l])
        print ("Absolut mean activation of labels:",allAbsmean/allCount)
        print ("Quadratic mean activation of labels:",allQmean/allCount)


    def calcHistogram(self):
        hist = {}
        allHist = []
        for i in range(len(self.data.sentences)):
            for j in range(len(self.data.sentences[i].words)):
                try:
                    l = self.data.sentences[i].labels[j]
                except AttributeError:
                    l = "none"
                h = self.data.sentences[i].data[j]
                if(l in hist):
                    hist[l] += h.tolist()
                else:
                    hist[l] = h.tolist()
                allHist += h.tolist()
        allHist.sort()
        for l in hist.keys():
            hist[l].sort()
            print("Histogramm for ",l,hist[l][0],end=" ")
            for i in range(10):
                index =  int((i+1.0)/10*len(hist[l]))-1
                print (hist[l][index],end=" ")
            print ("")
        print("Histogramm for all",allHist[0],end=" ")
        for i in range(10):
            index =  int((i+1.0)/10*len(allHist))-1
            print (allHist[index],end=" ")
        print ("")


    def calcMeanAndVariationWordVectors(self):
        mean = {}
        count = {}
        meanAll = zeros(self.data.sentences[0].data[0].shape)
        countAll = 0
        for i in range(len(self.data.sentences)):
            for j in range(len(self.data.sentences[i].words)):
                try:
                    l = self.data.sentences[i].labels[j]
                except AttributeError:
                    l = "none"
                h = self.data.sentences[i].data[j]
                if(l in mean):
                    mean[l] += h
                    count[l] += 1
                else:
                    mean[l] = copy(h)
                    count[l] = 1
                meanAll += h
                countAll += 1
        for k in mean.keys():
            mean[k] = mean[k]/count[k]
        meanAll = meanAll/countAll
        for k in mean.keys():
            for l in mean.keys():
                if(l != k):
                    diff = LA.norm(mean[k] - mean[l])
                    print ("Mean different of labels",l,"and",k,":",diff)
        print ("Mean vector:",LA.norm(meanAll))
        diff = {}
        diffAll = 0
        for i in range(len(self.data.sentences)):
            for j in range(len(self.data.sentences[i].words)):
                try:
                    l = self.data.sentences[i].labels[j]
                except AttributeError:
                    l = "none"
                h = self.data.sentences[i].data[j]
                d = LA.norm(h - mean[l])
                diffAll += LA.norm(h-meanAll)
                if(l in diff):
                    diff[l] += d
                else:
                    diff[l] = d
        
        for k in diff.keys():
            diff[k] = diff[k]/count[k]
            print("Variance in cluster",k,":",diff[k])
        print("Overall Variance:",diffAll/countAll)

def inspect(data):
    a = Intrinsic(data)
    return a.inspect()
