#!/usr/bin/env python

from __future__ import print_function
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils
from torch import IntTensor
from torch import FloatTensor
import torch
import deepdish as dd
from numpy import zeros


class Predictor:

    def __init__(self, data, model_file, load_model, store_model, input_type, output):
        self.data = data
        self.epochs = 10
        self.model_file = model_file
        self.load_model = load_model
        self.store_model = store_model
        self.input_type = input_type
        self.output = output

    def inspect(self):

        if self.load_model:
            self.load()
        self.prepareData()
        if not self.load_model:
            self.buildModel();
            self.trainloader = utils.data.DataLoader(self.dataset, batch_size=16,
                                                     shuffle=True)
            self.train()
        self.model.eval()
        self.trainloader = utils.data.DataLoader(self.dataset, batch_size=16,
                                                 shuffle=False)
        self.predict()
        if self.store_model:
            self.save()

    def load(self):
        param = dd.io.load(self.model_file + ".paramter.h5")
        self.inputSize = param['inputSize'].item()
        self.outputSize = param['outputSize'].item()
        self.model = nn.Sequential(nn.Linear(self.inputSize, self.outputSize))
        self.model.load_state_dict(torch.load(self.model_file + ".model"))


    def save(self):
        dd.io.save(self.model_file + ".paramter.h5", {'inputSize': self.inputSize,
                                                      'outputSize': self.outputSize},
                   compression=('blosc', 9))
        torch.save(self.model.state_dict(), self.model_file + ".model")

    def prepareData(self):

        samples = []
        ls = []

        for i in range(len(self.data.sentences)):
            if (self.input_type == "word"):
                for j in range(len(self.data.sentences[i].words)):
                    l = self.data.target_representation[i].data[j].tolist()

                    samples.append(self.data.sentences[i].data[j].tolist())
                    ls.append(l)
            elif (self.input_type == "sentence"):
                d = zeros(self.data.sentences[i].data[0].shape)
                for j in range(len(self.data.sentences[i].words)):
                    d += self.data.sentences[i].data[j]
                samples.append(d.tolist())
                l = zeros(self.data.target_representation[i].data[0].shape)
                for j in range(len(self.data.sentences[i].words)):
                    l += self.data.target_representation[i].data[j]
                ls.append(l.tolist())

        self.dataset = utils.data.TensorDataset(FloatTensor(samples), FloatTensor(ls))

    def buildModel(self):
        self.inputSize = self.data.sentences[0].data[0].size
        self.outputSize = self.data.target_representation[0].data[0].size
        self.model = nn.Sequential(nn.Linear(self.inputSize, self.outputSize))
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)

    def predict(self):

        all = 0;
        count = 0

        for i, data in enumerate(self.trainloader, 0):
            # get the inputs
            inputs, labels = data

            # forward + backward + optimize
            outputs = self.model(inputs.double())
            loss = labels.double()-outputs
            loss = loss.mul(loss)
            loss = loss.sum(1)
            all += loss.sum().item()
            count += loss.shape[0]
            if (self.output):
                for i in range(loss.shape[0]):
                    print("Distance: ", loss[i].item())

        print(" Distance: ", all, " Avg distance: ", 1.0 * all / count)

    def train(self):

        for epoch in range(self.epochs):  # loop over the dataset multiple times

            running_loss = 0.0
            for i, data in enumerate(self.trainloader, 0):
                # get the inputs
                inputs, labels = data

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(inputs.double())
                loss = self.criterion(outputs, labels.double())
                loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += loss.item();
                if i % 100 == 99:  # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / (i + 1)))

            print('epoch %d loss: %.3f' %
                  (epoch + 1, running_loss / (i + 1)))

        print('Finished Training')


def inspect(data, model_file, load_model, store_model, input_type, output):
    a = Predictor(data, model_file, load_model, store_model, input_type, output)
    return a.inspect()
