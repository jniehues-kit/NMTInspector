#!/usr/bin/env python

from __future__ import print_function
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils
from torch import IntTensor
from torch import FloatTensor
from torch.autograd import Variable
import torch
import deepdish as dd
from numpy import zeros


class Classifier:

    def __init__(self, data,model_file,load_model,store_model,input_type):
        self.data = data
        self.epochs = 10
        self.model_file = model_file
        self.load_model = load_model
        self.store_model = store_model
        self.input_type = input_type




    def inspect(self):

        if self.load_model:
            self.load()
        self.prepareData()
        if not self.load_model:
            self.buildModel();
            self.train()
        self.model.eval()
        self.predict()
        if self.store_model:
            self.save()

    def load(self):
        param = dd.io.load(self.model_file+".paramter.h5")
        self.mapping = param["mapping"]
        self.inputSize = param['inputSize'].item()
        self.outputSize = param['outputSize'].item()
        self.model = nn.Sequential(nn.Linear(self.inputSize,self.outputSize))
        self.model.load_state_dict(torch.load(self.model_file+".model"))


    def save(self):
        dd.io.save(self.model_file+".paramter.h5", {'mapping': self.mapping,
                                                'inputSize': self.inputSize,
                                                'outputSize':self.outputSize},
                                  compression=('blosc', 9))
        torch.save(self.model.state_dict(), self.model_file+".model")



    def prepareData(self):

        samples = []
        labels = []
        self.mapping = {}

        for i in range(len(self.data.sentences)):
            if(self.input_type == "word"):
                for j in range(len(self.data.sentences[i].words)):
                    try:
                        l = self.data.sentences[i].labels[j]
                    except AttributeError:
                        l = "none"

                    if l in self.mapping:
                        c = self.mapping[l]
                    else:
                        c = len(self.mapping)
                        self.mapping[l] = c

                    samples.append(self.data.sentences[i].data[j].tolist())
                    labels.append(c)
            elif(self.input_type == "sentence"):
                d = zeros(self.data.sentences[i].data[0].shape)
                for j in range(len(self.data.sentences[i].words)):
                    d += self.data.sentences[i].data[j]
                samples.append(d.tolist())
                try:
                    l = self.data.sentences[i].label
                except AttributeError:
                    l = "none"

                if l in self.mapping:
                    c = self.mapping[l]
                else:
                    c = len(self.mapping)
                    self.mapping[l] = c
                labels.append(c)

        data = utils.data.TensorDataset(FloatTensor(samples), IntTensor(labels))
        self.trainloader = utils.data.DataLoader(data, batch_size=16,
                                                  shuffle=True)

    def buildModel(self):
        self.inputSize = self.data.sentences[0].data[0].size
        self.outputSize = len(self.mapping)
        self.model = nn.Sequential(nn.Linear(self.inputSize,self.outputSize))
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.0001)


    def predict(self):

        correct = 0;
        all = 0;

        for i, data in enumerate(self.trainloader, 0):
            # get the inputs
            inputs, labels = data


            # forward + backward + optimize
            #outputs = self.model(inputs)
            #loss = self.criterion(outputs, labels)
            outputs = self.model(Variable(inputs))
            top_n, top_i = outputs.topk(1)
            correct += top_i.view(labels.size()).eq(Variable(labels)).sum().data[0]
            all += labels.numel();

        print(correct," of ",all,"elements correct: ",1.0*correct/all)

    def train(self):

        for epoch in range(self.epochs):  # loop over the dataset multiple times

            running_loss = 0.0
            for i, data in enumerate(self.trainloader, 0):
                # get the inputs
                inputs, labels = data

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                #outputs = self.model(inputs)
                #loss = self.criterion(outputs, labels)
                outputs = self.model(Variable(inputs))
                loss = self.criterion(outputs, Variable(labels))
                loss.backward()
                self.optimizer.step()

                # print statistics
                #running_loss += loss.item()
                running_loss += loss;
                if i % 100 == 99:  # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / (i+1)))

            print('epoch %d loss: %.3f' %
                (epoch + 1, running_loss / (i+1)))

        print('Finished Training')



def inspect(data,model_file,load_model,store_model,input_type):
    a = Classifier(data,model_file,load_model,store_model,input_type)
    return a.inspect()
