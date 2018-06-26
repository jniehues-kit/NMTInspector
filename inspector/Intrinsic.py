#!/usr/bin/env python


import sys
from numpy import linalg as LA

class Intrinsic:

    def __init__(self,data):
        self.data = data

    def inspect(self):
        self.calcMeanAndVariationWordVectors();
        self.calcMeanActiation();
        self.calcEntropyActiation();
        return []

    def calcEntropyActivation(self):
        entropy = {}
        count = {}
        for i in range(len(self.data.sentences)):
            for j in range(len(self.data.sentences[i].words)):
                l = self.data.sentences[i].labels[j]
                h = self.data.sentences[i].data[j]
                if(l in mean):
                    entropy[l] += np.dot(h,log(h))
                    count[l] += h.size()
                else:
                    entropy[l] = np.dot(h,log(h))
                    count[l] = h.size()
        allEntropy = 0
        allCount = 0
        for k in mean.keys():
            allEntropy += entropy[l]
            print ("Entropy of activation of labels",l,":",entropy[l]/count[l])
        print ("Entropy of activation of labels:",allEntropy[l]/allCount[l])



    def calcMeanActivation(self):
        absmean = {}
        qmean = {}
        count = {}
        for i in range(len(self.data.sentences)):
            for j in range(len(self.data.sentences[i].words)):
                l = self.data.sentences[i].labels[j]
                h = self.data.sentences[i].data[j]
                if(l in mean):
                    abnsmean[l] += sum(abs(h))
                    qmean[l] += np.dot(h,h)
                    count[l] += h.size()
                else:
                    absmean[l] = sum(abs(h))
                    qmean[l] = np.dot(h,h)
                    count[l] = h.size()
        allAbsmean = 0
        allQmean = 0
        allCount = 0
        for k in mean.keys():
            allAbsmean += absmean[l]
            allQmean += qmean[l]
            allCount += count
            print ("Absolut mean activation of labels",l,":",absmean[l]/count[l])
            print ("Quadratic mean activation of labels",l,":",qmean[l]/count[l])
        print ("Absolut mean activation of labels:",allAbsmean[l]/allCount[l])
        print ("Quadratic mean activation of labels:",allQmean[l]/allCount[l])




    def calcMeanAndVariationWordVectors(self):
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
                    mean[l] = h
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
