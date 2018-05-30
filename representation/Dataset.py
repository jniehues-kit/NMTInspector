#!/usr/bin/env python


import sys

class Dataset:
    def __init__(self,label):
        self.label = label
        self.sentences = []

class Sentence:
    def __init__(self,rep):
        self.data = rep;
        self.label = ""
        #self.word_label = [""] * self.data.size[0]
