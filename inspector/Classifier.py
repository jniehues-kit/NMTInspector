#!/usr/bin/env python

from __future__ import print_function
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils
from torch import IntTensor
from torch import FloatTensor
from torch.autograd import Variable

class Classifier:

    def __init__(self, data):
        self.data = data
        self.epochs = 10




    def inspect(self):

        self.prepareData()
        self.buildModel();
        self.train()
        self.predict()
        #self.save()

    def prepareData(self):

        samples = []
        labels = []
        self.mapping = {}

        for i in range(len(self.data.sentences)):
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

        data = utils.data.TensorDataset(FloatTensor(samples), IntTensor(labels))
        self.trainloader = utils.data.DataLoader(data, batch_size=16,
                                                  shuffle=True)

    def buildModel(self):
        inputSize = self.data.sentences[0].data[0].size
        outputSize = len(self.mapping)
        self.model = nn.Sequential(nn.Linear(inputSize,outputSize))
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.0001)


    def predict(self):


        for i, data in enumerate(self.trainloader, 0):
            # get the inputs
            inputs, labels = data


            # forward + backward + optimize
            #outputs = self.model(inputs)
            #loss = self.criterion(outputs, labels)
            outputs = self.model(Variable(inputs))
            top_n, top_i = outputs.topk(1)
            #print ("Labels:",labels)
            #print ("Prediction:",outputs)


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



def inspect(data):
    a = Classifier(data)
    return a.inspect()
