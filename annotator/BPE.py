#!/usr/bin/env python


import sys


class BPEAnnotator:

    def __init__(self,data):
        self.data = data

    def annotate(self):
        print ("Annotate")
        for i in range(len(self.data.sentences)):
            labels = []
            for j in range(len(self.data.sentences[i].words)):
                if(self.data.sentences[i].words[j][-2:] == "@@"):
                    if(j == 0):
                        labels.append("S")
                    elif(self.data.sentences[i].words[j-1][-2:] == "@@"):
                        labels.append("I")
                    else:
                        labels.append("S")
                elif( j == 0):
                    labels.append("N")
                elif(self.data.sentences[i].words[j-1][-2:] == "@@"):
                    labels.append("E")
                else:
                    labels.append("N")
            self.data.sentences[i].labels =labels
            #print ("Hidden:",self.data.sentences[i].data.shape)
            #print ("Sentences:", " ".join(self.data.sentences[i].words))
            #print ("Labels:", " ".join(self.data.sentences[i].labels))
            
def annotate(data):
    a = BPEAnnotator(data)
    a.annotate()
