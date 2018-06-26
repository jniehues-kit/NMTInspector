#!/usr/bin/env python


import sys
import argparse

import torch
from torch.autograd import Variable

from onmt.translate.Translator import make_translator
import onmt.opts
import onmt.translate.Translator

import representation.Dataset

class InpsectTranslator(onmt.translate.Translator):

    def init_representation(self,label):
        self.rep = representation.Dataset.Dataset(label)

        
    def translate_batch(self, batch, data):
        """
        Copied from OpenNMT -> only change is that we store the hidden representations
        Translate a batch of sentences.

        Mostly a wrapper around :obj:`Beam`.

        Args:
           batch (:obj:`Batch`): a batch from a dataset object
           data (:obj:`Dataset`): the dataset object


        Todo:
           Shouldn't need the original dataset.
        """

        # (0) Prep each of the components of the search.
        # And helper method for reducing verbosity.
        beam_size = self.beam_size
        batch_size = batch.batch_size
        data_type = data.data_type
        vocab = self.fields["tgt"].vocab

        # Define a list of tokens to exclude from ngram-blocking
        # exclusion_list = ["<t>", "</t>", "."]
        exclusion_tokens = set([vocab.stoi[t]
                                for t in self.ignore_when_blocking])

        beam = [onmt.translate.Beam(beam_size, n_best=self.n_best,
                                    cuda=self.cuda,
                                    global_scorer=self.global_scorer,
                                    pad=vocab.stoi[onmt.io.PAD_WORD],
                                    eos=vocab.stoi[onmt.io.EOS_WORD],
                                    bos=vocab.stoi[onmt.io.BOS_WORD],
                                    min_length=self.min_length,
                                    stepwise_penalty=self.stepwise_penalty,
                                    block_ngram_repeat=self.block_ngram_repeat,
                                    exclusion_tokens=exclusion_tokens)
                for __ in range(batch_size)]

        # Help functions for working with beams and batches
        def var(a): return Variable(a, volatile=True)

        def rvar(a): return var(a.repeat(1, beam_size, 1))

        def bottle(m):
            return m.view(batch_size * beam_size, -1)

        def unbottle(m):
            return m.view(beam_size, batch_size, -1)

        # (1) Run the encoder on the src.

        src = onmt.io.make_features(batch, 'src', data_type)
        src_lengths = None
        if data_type == 'text':
            _, src_lengths = batch.src

        enc_states, memory_bank = self.model.encoder(src, src_lengths)
        for i in range(memory_bank.size(1)):
            s = representation.Dataset.Sentence(memory_bank.select(1,i).narrow(0,0,batch.src[1][i]).data.numpy())
            words = []
            for j in range(batch.src[1][i]):
                words.append(self.fields["src"].vocab.itos[batch.src[0].data[j][i]])
            s.words = words;
            while(len(self.rep.sentences) <= batch.indices.data[i]):
                self.rep.sentences.append(None)
            self.rep.sentences[batch.indices.data[i]] = s

        dec_states = self.model.decoder.init_decoder_state(
            src, memory_bank, enc_states)

        if src_lengths is None:
            src_lengths = torch.Tensor(batch_size).type_as(memory_bank.data)\
                                                  .long()\
                                                  .fill_(memory_bank.size(0))

        # (2) Repeat src objects `beam_size` times.
        src_map = rvar(batch.src_map.data) \
            if data_type == 'text' and self.copy_attn else None
        memory_bank = rvar(memory_bank.data)
        memory_lengths = src_lengths.repeat(beam_size)
        dec_states.repeat_beam_size_times(beam_size)

        # (3) run the decoder to generate sentences, using beam search.
        for i in range(self.max_length):
            if all((b.done() for b in beam)):
                break

            # Construct batch x beam_size nxt words.
            # Get all the pending current beam words and arrange for forward.
            inp = var(torch.stack([b.get_current_state() for b in beam])
                      .t().contiguous().view(1, -1))

            # Turn any copied words to UNKs
            # 0 is unk
            if self.copy_attn:
                inp = inp.masked_fill(
                    inp.gt(len(self.fields["tgt"].vocab) - 1), 0)

            # Temporary kludge solution to handle changed dim expectation
            # in the decoder
            inp = inp.unsqueeze(2)

            # Run one step.
            dec_out, dec_states, attn = self.model.decoder(
                inp, memory_bank, dec_states, memory_lengths=memory_lengths)
            dec_out = dec_out.squeeze(0)
            # dec_out: beam x rnn_size

            # (b) Compute a vector of batch x beam word scores.
            if not self.copy_attn:
                out = self.model.generator.forward(dec_out).data
                out = unbottle(out)
                # beam x tgt_vocab
                beam_attn = unbottle(attn["std"])
            else:
                out = self.model.generator.forward(dec_out,
                                                   attn["copy"].squeeze(0),
                                                   src_map)
                # beam x (tgt_vocab + extra_vocab)
                out = data.collapse_copy_scores(
                    unbottle(out.data),
                    batch, self.fields["tgt"].vocab, data.src_vocabs)
                # beam x tgt_vocab
                out = out.log()
                beam_attn = unbottle(attn["copy"])
            # (c) Advance each beam.
            for j, b in enumerate(beam):
                b.advance(out[:, j],
                          beam_attn.data[:, j, :memory_lengths[j]])
                dec_states.beam_update(j, b.get_current_origin(), beam_size)

        # (4) Extract sentences from beam.
        ret = self._from_beam(beam)
        ret["gold_score"] = [0] * batch_size
        if "tgt" in batch.__dict__:
            ret["gold_score"] = self._run_target(batch, data)
        ret["batch"] = batch
        return ret


class ONMTGenerator:


    
    def __init__(self,model,src,representation,gpuid):
        self.model = model;
        self.representation = representation
        self.src = src
        self.gpuid = gpuid

        dummy_parser = argparse.ArgumentParser(description='train.py')
        onmt.opts.model_opts(dummy_parser)
        onmt.opts.translate_opts(dummy_parser)
        if(gpuid != ""):
            self.opt = dummy_parser.parse_known_args(["-model",self.model,"-src",self.src,"-gpuid",self.gpuid])[0]
        else:
            self.opt = dummy_parser.parse_known_args(["-model",self.model,"-src",self.src])[0]            
        
        print ("GPU:",self.opt.gpu)
        self.translator = make_translator(self.opt)
        self.translator.__class__ = InpsectTranslator
        self.translator.init_representation("testdata")
        
    def generate(self):
        self.translator.translate(self.opt.src_dir,self.opt.src,self.opt.tgt,self.opt.batch_size,self.opt.attn_debug)
        return self.translator.rep



def generate(source_test_data,model,representation,gpuid):
    g = ONMTGenerator(model,source_test_data,representation,gpuid)
    return g.generate()
