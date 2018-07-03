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

class ONMTGenerator:

    def __init__(self, model, src, representation, gpuid):
        self.model = model;
        self.representation = representation
        self.src = src
        self.gpuid = gpuid

        dummy_parser = argparse.ArgumentParser(description='train.py')
        onmt.opts.model_opts(dummy_parser)
        onmt.opts.translate_opts(dummy_parser)
        if (gpuid != ""):
            self.opt = dummy_parser.parse_known_args(["-model", self.model, "-src", self.src, "-gpu", self.gpuid])[0]
        else:
            self.opt = dummy_parser.parse_known_args(["-model", self.model, "-src", self.src])[0]

        self.translator = build_translator(self.opt)

        self.translator.model.encoder._vivisect = {"iteration": 0, "model_name": "OpenNMT", "framework": "pytorch"}

        probe(self.translator.model.encoder, "localhost", 8080, monitorONMT, performONMT)


        self.collector = Collector(self.representation);
        #self.translator.init_representation("testdata")


    def generate(self):
        #self.startCollector()
        t = Thread(target=startCollector, args=(self.collector,))
        t.start()
        self.translator.translate(src_path=self.opt.src,
                             tgt_path=self.opt.tgt,
                             src_dir=self.opt.src_dir,
                             batch_size=1,
                             attn_debug=self.opt.attn_debug)
        clear("localhost", 8080)
        return self.collector.data

def monitorONMT(layer):
    if(type(layer).__name__ == "LSTM"):
        return True
    return False

def performONMT(model, op, inputs, outputs):
    return True

def startCollector(collector):
    collector.run(port=8080,debug=False)



def generate(source_test_data, model, representation, gpuid):
    g = ONMTGenerator(model, source_test_data, representation, gpuid)
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

                self.storeData(j["outputs"])
                return "OK"

    def storeData(self,data):
        if(self.representation == "EncoderHiddenLayer"):
            lstm = numpy.array(data[0]);
            s = representation.Dataset.Sentence(lstm)
            s.words = []
            for i in range(len(lstm)):
                s.words.append("UNK")
            self.data.sentences.append(s)
