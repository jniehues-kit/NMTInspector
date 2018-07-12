#!/usr/bin/env python

import argparse
from sockeye import inference
from sockeye import score
from sockeye import constants as C
from contextlib import ExitStack
from sockeye import output_handler
from sockeye import arguments
from sockeye.translate import _setup_context

from vivisect.pytorch import probe

import representation.Dataset

import numpy

class SockeyeGenerator:

    def __init__(self, model, src,tgt,  rep, gpuid):
        self.model = model;
        print("Model:",self.model)
        self.representation = rep
        self.src = src
        self.tgt = tgt;
        self.gpuid = gpuid

        params = argparse.ArgumentParser(description='Scoring CLI')
        arguments.add_score_cli_args(params)
        param = ["-m",self.model]
        if (gpuid == ""):
            param += ["--use-cpu"]

        print(param)
        args = params.parse_known_args(param)[0]

        #dummy_parser = argparse.ArgumentParser(description='train.py')
        #onmt.opts.model_opts(dummy_parser)
        #onmt.opts.translate_opts(dummy_parser)
        #param = ["-model", self.model, "-src", self.src]

        #if (gpuid != ""):
        #    param += ["-gpu", self.gpuid]
        #if (self.tgt != ""):
        #    param += ["-tgt",self.tgt]

        #self.opt = dummy_parser.parse_known_args(param)[0]

        #self.translator = build_translator(self.opt)



        self.output_handler = output_handler.get_output_handler(C.OUTPUT_HANDLER_TRANSLATION_WITH_SCORE,
                                            None,
                                            0.9)

        with ExitStack() as exit_stack:
            context = _setup_context(args, exit_stack)

            models, source_vocabs, target_vocab = inference.load_models(
                context=context,
                max_input_len=None,
                beam_size=1,
                batch_size=1,
                model_folders=[self.model],
                checkpoints=None,
                softmax_temperature=None,
                max_output_length_num_stds=C.DEFAULT_NUM_STD_MAX_OUTPUT_LENGTH,
                decoder_return_logit_inputs=False,
                cache_output_layer_w_b=False)
            self.translator = inference.Translator(context=context,
                                              ensemble_mode="linear",
                                              bucket_source_width=10,
                                              length_penalty=inference.LengthPenalty(1.0,0.0),
                                              beam_prune=0,
                                              beam_search_stop=C.BEAM_SEARCH_STOP_ALL,
                                              models=models,
                                              source_vocabs=source_vocabs,
                                              target_vocab=target_vocab,
                                              restrict_lexicon=None,
                                              store_beam=False,
                                              strip_unknown_words=False)


        self.data = representation.Dataset.Dataset("testdata")



        #if(self.representation == "EncoderWordEmbeddings" or self.representation == "EncoderHiddenLayer"):
        #    self.translator.model.encoder._vivisect = {"iteration":0, "rescore": 1, "model_name": "OpenNMT", "framework": "pytorch"}
        #    probe(self.translator.model.encoder, select=self.monitorONMT, perform=self.performONMT,cb=self.storeData)
        #elif(self.representation == "ContextVector" or self.representation == "DecoderWordEmbeddings" or self.representation == "DecoderHiddenLayer"):
        #    #need to use the encoder to see when a sentence start
        #    self.translator.model.decoder._vivisect = {"iteration":0, "sentence": 0, "model_name": "OpenNMT", "framework": "pytorch"}
        #    probe(self.translator.model.decoder,select=self.monitorONMT, perform=self.performONMT,cb=self.storeData)
        #    self.translator.model.encoder._vivisect = {"iteration":0, "rescore": 1, "model_name": "OpenNMT", "framework": "pytorch"}
        #    probe(self.translator.model.encoder,select=self.monitorONMT, perform=self.performONMT,cb=self.storeData)
        #else:
        #    print("Unkown representation:",self.representation)





    def generate(self):
        print("Target file:",self.tgt)
        score.read_and_translate(translator=self.translator,
                       output_handler=self.output_handler,
                       chunk_size=None,
                       input_file=self.src,
                       target_file=self.tgt,
                       input_factors=None,
                       input_is_json=False)
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
    g = SockeyeGenerator(model, source_test_data, target_test_data, representation, gpuid)
    return g.generate()






