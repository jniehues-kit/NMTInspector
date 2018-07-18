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


class Autoencoder:

    def __init__(self, data, model_file, load_model, store_model, input_type, output,hiddenSize):
        self.data = data
        self.epochs = 10
        self.model_file = model_file
        self.load_model = load_model
        self.store_model = store_model
        self.input_type = input_type
        self.output = output
        self.hiddenSize = int(hiddenSize)

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
        self.hiddenSize = param['hiddenSize'].item()
        self.model = nn.Sequential(nn.Linear(self.inputSize, self.hiddenSize),
                                   nn.Sigmoid(),
                                   nn.Linear(self.hiddenSize,self.inputSize),
                                   nn.Sigmoid())
        self.model.load_state_dict(torch.load(self.model_file + ".model"))


    def save(self):
        dd.io.save(self.model_file + ".paramter.h5", {'inputSize': self.inputSize,
                                                      'hiddenSize': self.hiddenSize},
                   compression=('blosc', 9))
        torch.save(self.model.state_dict(), self.model_file + ".model")

    def prepareData(self):

        samples = []
        ls = []

        for i in range(len(self.data.sentences)):
            if (self.input_type == "word"):
                for j in range(len(self.data.sentences[i].words)):
                    samples.append(self.data.sentences[i].data[j].tolist())
            elif (self.input_type == "sentence"):
                d = zeros(self.data.sentences[i].data[0].shape)
                for j in range(len(self.data.sentences[i].words)):
                    d += self.data.sentences[i].data[j]
                samples.append(d.tolist())

        self.dataset = utils.data.TensorDataset(FloatTensor(samples))

    def buildModel(self):
        self.inputSize = self.data.sentences[0].data[0].size
        self.model = nn.Sequential(nn.Linear(self.inputSize, self.hiddenSize),
                                   nn.Sigmoid(),
                                   nn.Linear(self.hiddenSize,self.inputSize),
                                   nn.Sigmoid())
        self.criterion = nn.MSELoss(size_average=False)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)

    def predict(self):

        all = 0;
        count = 0
        relative = 0;

        for i, data in enumerate(self.trainloader, 0):
            # get the inputs
            inputs = data[0]

            # forward + backward + optimize
            outputs = self.model(inputs.double())
            loss = inputs.double()-outputs
            loss = loss.mul(loss)
            loss = loss.sum(1)
            length = outputs.mul(outputs).sum(1)
            all += loss.sum().item()
            relative += (100*loss/length).sum().item()
            count += loss.shape[0]
            if (self.output):
                for i in range(loss.shape[0]):
                    print("Distance: ", loss[i].item(), "Length: ",length[i].item(),"Relativ distance: ",100*loss[i].item()/length[i].item())

        print(" Distance: ", all, " Avg distance: ", 1.0 * all / count)
        print(" rel. Distance: ", relative, " Avg rel distance: ", 1.0 * relative / count)

    def train(self):

        for epoch in range(self.epochs):  # loop over the dataset multiple times

            running_loss = 0.0
            for i, data in enumerate(self.trainloader, 0):
                # get the inputs
                inputs = data[0]

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(inputs.double())
                loss = self.criterion(outputs, inputs.double())
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


def inspect(data, model_file, load_model, store_model, input_type, output,param):
    a = Autoencoder(data, model_file, load_model, store_model, input_type, output,param)
    return a.inspect()
