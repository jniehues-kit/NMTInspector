#!/usr/bin/env python

import argparse

from onmt.translate.translator import build_translator
import onmt.opts
import onmt.translate.translator

from vivisect.pytorch import probe
from vivisect.servers import clear
from threading import Thread

import representation.Dataset
from flask import Flask, request

import numpy
import sys

class ONMTGenerator:

    def __init__(self, model, src,tgt,  representation, gpuid):
        self.model = model;
        self.representation = representation
        self.src = src
        self.tgt = tgt;
        self.gpuid = gpuid
        self.port = 8882

        dummy_parser = argparse.ArgumentParser(description='train.py')
        onmt.opts.model_opts(dummy_parser)
        onmt.opts.translate_opts(dummy_parser)
        param = ["-model", self.model, "-src", self.src]

        if (gpuid != ""):
            param += ["-gpu", self.gpuid]
        if (self.tgt != ""):
            param += ["-tgt",self.tgt]

        self.opt = dummy_parser.parse_known_args(param)[0]

        self.translator = build_translator(self.opt)



        if(self.representation == "EncoderWordEmbeddings" or self.representation == "EncoderHiddenLayer"):
            self.translator.model.encoder._vivisect = {"iteration":0, "rescore": 1, "sentence": 0,"model_name": "OpenNMT", "framework": "pytorch"}
            probe(self.translator.model.encoder, "localhost", self.port, self.monitorONMT, self.performONMT)
        elif(self.representation == "ContextVector" or self.representation == "DecoderWordEmbeddings" or self.representation == "DecoderHiddenLayer"):
            #need to use the encoder to see when a sentence start
            self.translator.model.decoder._vivisect = {"iteration":0, "sentence": 0, "model_name": "OpenNMT", "framework": "pytorch"}
            probe(self.translator.model.decoder, "localhost", self.port, self.monitorONMT, self.performONMT)
            self.translator.model.encoder._vivisect = {"iteration":0, "rescore": 1, "model_name": "OpenNMT", "framework": "pytorch"}
            probe(self.translator.model.encoder, "localhost", self.port, self.monitorONMT, self.performONMT)
        else:
            print("Unkown representation:",self.representation)



        self.collector = Collector(self.representation);
        #self.translator.init_representation("testdata")


    def generate(self):
        #self.startCollector()
        t = Thread(target=startCollector, args=(self.collector,self.port,))
        t.start()
        self.translator.translate(src_path=self.opt.src,
                             tgt_path=self.opt.tgt,
                             src_dir=self.opt.src_dir,
                             batch_size=1,
                             attn_debug=self.opt.attn_debug)
        clear("localhost", self.port)
        print(len(self.collector.data.sentences))
        return self.collector.data

    def monitorONMT(self,layer):
        if(type(layer).__name__ == "LSTM" and self.representation == "EncoderHiddenLayer"):
            return True
        elif (type(layer).__name__ == "Embeddings" and self.representation == "EncoderWordEmbeddings"):
            return True
        elif (type(layer).__name__ == "GlobalAttention" and self.representation == "ContextVector"):
            return True
        elif (type(layer).__name__ == "StackedLSTM" and self.representation == "DecoderHiddenLayer"):
            return True
        elif (type(layer).__name__ == "Embeddings" and self.representation == "DecoderWordEmbeddings"):
            return True
        elif (type(layer).__name__ == "LSTM" and (self.representation == "ContextVector" or
        self.representation == "DecoderHiddenLayer")):
            return True
        return False

    def performONMT(self,model, op, inputs, outputs):
        if type(model).__name__ == "RNNEncoder":
            self.translator.model.encoder._vivisect["rescore"] = 1 - self.translator.model.encoder._vivisect["rescore"]
            if(self.representation == "EncoderHiddenLayer" or self.representation == "EncoderWordEmbeddings"):
                if(self.tgt != ""):
                    if self.translator.model.encoder._vivisect["rescore"] == 1:
                        self.translator.model.encoder._vivisect["sentence"] += 1
                        if(self.translator.model.encoder._vivisect["sentence"] % 10 == 0):
                            print("Starting sentences: ",self.translator.model.encoder._vivisect["sentence"],file=sys.stderr)
                    #if target is given it will do rescoring and translation -> only use every second
                    return self.translator.model.encoder._vivisect["rescore"] == 1
                else:
                    return True
            if(self.representation == "ContextVector" or self.representation == "DecoderWordEmbeddings" or
                    self.representation == "DecoderHiddenLayer"):
                if self.translator.model.encoder._vivisect["rescore"] == 1:
                    #need to know which sentence ends
                    self.translator.model.decoder._vivisect["sentence"] += 1
                    if(self.translator.model.decoder._vivisect["sentence"] % 10 == 0):
                        print("Starting sentences: ",self.translator.model.decoder._vivisect["sentence"],file=sys.stderr)
                return False
        elif type(model).__name__ == "InputFeedRNNDecoder":
            # if target is given it will do rescoring and translation -> only use every second
            return self.translator.model.encoder._vivisect["rescore"] == 1


def startCollector(collector,port):
    collector.run(port=port,debug=False)



def generate(source_test_data, target_test_data, model, representation, gpuid):
    g = ONMTGenerator(model, source_test_data, target_test_data, representation, gpuid)
    return g.generate()






class Collector(Flask):

    def __init__(self,rep):
        super(Collector, self).__init__("Collector")
        self.data = representation.Dataset.Dataset("testdata")
        self.representation = rep

        @self.route("/clear", methods=["POST"])
        def clear():
            func = request.environ.get('werkzeug.server.shutdown')
            if func is None:
                raise RuntimeError('Not running with the Werkzeug Server')
            func()
            return "OK"

        @self.route("/", methods=["GET", "POST"])
        def handle():
            if request.method == "GET":
                return "Aggregator server"
            elif request.method == "POST":
                j = request.get_json()

                self.storeData(j["outputs"],j["metadata"])
                return "OK"

    def storeData(self,data,meta):
        if(self.representation == "EncoderHiddenLayer"):
            lstm = numpy.array(data[0]);
            s = representation.Dataset.Sentence(lstm)
            s.words = []
            for i in range(len(lstm)):
                s.words.append("UNK")
            self.data.sentences.append(s)
        elif(self.representation == "EncoderWordEmbeddings"):
            emb = numpy.array(data).squeeze()
            s = representation.Dataset.Sentence(emb)
            s.words = []
            for i in range(len(emb)):
                s.words.append("UNK")
            self.data.sentences.append(s)
        elif(self.representation == "ContextVector"):
            cv = numpy.array(data[0]).squeeze()
            if(meta["sentence"] != len(self.data.sentences)):
                e = numpy.array([])
                e.resize(0,cv.shape[0])
                self.data.sentences.append(representation.Dataset.Sentence(e))
                self.data.sentences[-1].words = []
            self.data.sentences[-1].addWord(cv,"UNK")

        elif(self.representation == "DecoderWordEmbeddings"):
            emb = numpy.array(data).squeeze()
            s = representation.Dataset.Sentence(emb)
            s.words = []
            for i in range(len(emb)):
                s.words.append("UNK")
            self.data.sentences.append(s)
        elif(self.representation == "DecoderHiddenLayer"):
            cv = numpy.array(data[0]).squeeze()
            if(meta["sentence"] != len(self.data.sentences)):
                e = numpy.array([])
                e.resize(0,cv.shape[0])
                self.data.sentences.append(representation.Dataset.Sentence(e))
                self.data.sentences[-1].words = []
            self.data.sentences[-1].addWord(cv,"UNK")
