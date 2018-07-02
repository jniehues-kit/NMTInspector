#!/usr/bin/env python

import argparse

from onmt.translate.Translator import make_translator
import onmt.opts
import onmt.translate.Translator

from vivisect.pytorch import probe

import representation.Dataset


def monitorONMT(layer):
    print ("Layer:",layer)
    return True

def performONMT(model, op, inputs, outputs):
    return True


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

        self.translator = make_translator(self.opt)

        probe(self.translator.model.encoder, "0.0.0.0", 39628, monitorONMT, performONMT)

        #self.translator.init_representation("testdata")

    def generate(self):
        self.translator.translate(self.opt.src_dir, self.opt.src, self.opt.tgt, self.opt.batch_size,
                                  self.opt.attn_debug)
        return self.translator.rep


def generate(source_test_data, model, representation, gpuid):
    g = ONMTGenerator(model, source_test_data, representation, gpuid)
    return g.generate()


