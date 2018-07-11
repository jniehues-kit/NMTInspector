#!/usr/bin/env python

import argparse

from onmt.translate.translator import build_translator
import onmt.opts
import onmt.translate.translator

from vivisect.pytorch import probe

import representation.Dataset

import numpy

class ONMTGenerator:

    def __init__(self, model, src,tgt,  rep, gpuid):
        self.model = model;
        self.representation = rep
        self.src = src
        self.tgt = tgt;
        self.gpuid = gpuid

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
        self.data = representation.Dataset.Dataset("testdata")



        if(self.representation == "EncoderWordEmbeddings" or self.representation == "EncoderHiddenLayer"):
            self.translator.model.encoder._vivisect = {"iteration":0, "rescore": 1, "model_name": "OpenNMT", "framework": "pytorch"}
            probe(self.translator.model.encoder, select=self.monitorONMT, perform=self.performONMT,cb=self.storeData)
        elif(self.representation == "ContextVector" or self.representation == "DecoderWordEmbeddings" or self.representation == "DecoderHiddenLayer"):
            #need to use the encoder to see when a sentence start
            self.translator.model.decoder._vivisect = {"iteration":0, "sentence": 0, "model_name": "OpenNMT", "framework": "pytorch"}
            probe(self.translator.model.decoder,select=self.monitorONMT, perform=self.performONMT,cb=self.storeData)
            self.translator.model.encoder._vivisect = {"iteration":0, "rescore": 1, "model_name": "OpenNMT", "framework": "pytorch"}
            probe(self.translator.model.encoder,select=self.monitorONMT, perform=self.performONMT,cb=self.storeData)
        else:
            print("Unkown representation:",self.representation)





    def generate(self):
        self.translator.translate(src_path=self.opt.src,
                             tgt_path=self.opt.tgt,
                             src_dir=self.opt.src_dir,
                             batch_size=1,
                             attn_debug=self.opt.attn_debug)
        print(len(self.data.sentences))
        return self.data

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
                    #if target is given it will do rescoring and translation -> only use every second
                    return self.translator.model.encoder._vivisect["rescore"] == 1
                else:
                    return True
            if(self.representation == "ContextVector" or self.representation == "DecoderWordEmbeddings" or
                    self.representation == "DecoderHiddenLayer"):
                if self.translator.model.encoder._vivisect["rescore"] == 1:
                    #need to know which sentence ends
                    self.translator.model.decoder._vivisect["sentence"] += 1
                return False
        elif type(model).__name__ == "InputFeedRNNDecoder":
            # if target is given it will do rescoring and translation -> only use every second
            return self.translator.model.encoder._vivisect["rescore"] == 1

    def storeData(self,context):
        data = context["outputs"]
        meta = context["metadata"]
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




def generate(source_test_data, target_test_data, model, representation, gpuid):
    g = ONMTGenerator(model, source_test_data, target_test_data, representation, gpuid)
    return g.generate()






