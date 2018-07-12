#!/usr/bin/env python


import sys

import toolkits.ONMT
import toolkits.vivisectONMT
import toolkits.vivisectSockeye
import argparse
import logging
#import representation.DataSet
import annotator.BPE
import annotator.SentenceLabels
import annotator.TokenLabels
import inspector.Intrinsic
import inspector.Classifier
import json



logging.basicConfig(format='%(message)s')


def main():

    parser = argparse.ArgumentParser(description='Inspect.py')
    
    
    parser.add_argument('-prediction_task', type=str,default="",
                        choices=['BPE','SentenceLabels','TokenLabels'],
                        help="""Predicition task to be used to analyse the hidden representations""")

    parser.add_argument('-ml_technique', type=str, default="intrinsic",
                        choices=['intrinsic','classifier','squeuncePrediction','unsupervised','none'],
                        help="""Technique used to analse the hidden representations""")

    parser.add_argument('-inspection_model', type=str, default="",
                        help="""File to save/load the model used for inspection""")
    parser.add_argument('-store_inspection_model', action='store_true',
                       help="""Store the model used for inpsection""")
    parser.add_argument('-load_inspection_model', action='store_true',
                       help="""Load the model used for inpsection""")
    parser.add_argument('-output_predictions', action='store_true',
                       help="""Output class prediction for all sentences""")
    parser.add_argument('-inspection_model_input', type=str, default="Word",
                        choices=['word','sentence'],
                        help="""Use inspection model to predict:\n
                        Word: One label per word\n
                        Sentence: One label per word ( take sum of word representations )""")

    parser.add_argument('-representation', type=str, default="EncoderWordEmbeddings",
                        choices=['EncoderWordEmbeddings','EncoderHiddenLayer','ContextVector','DecoderWordEmbeddings',
                                 'DecoderHiddenLayer'],
                        help="""EncoderWordEmbeddings""")
    
    parser.add_argument('-source_test_data', type=str, default="",
                        help="""Path to the input data""")

    parser.add_argument('-target_test_data', type=str, default="",
                        help="""Path to the target data""")

    parser.add_argument('-model', type=str, default="",
                        help="""Model that should be analyzed""")

    parser.add_argument('-model_type', type=str, default="",
                        choices=['OpenNMT','vivisectONMT','vivisectSockeye'],
                        help="""Framework used to train the model""")


    parser.add_argument('-hidden_representation', type=str, default="",
                        help="""JSON File containing hidden representation from previous run""")

    parser.add_argument('-hidden_representation_out', type=str, default="",
                        help="""CSV File to store the hidden representation""")

    parser.add_argument('-label_file', type=str, default="",
                        help="""Label File""")


    # GPU
    parser.add_argument('-gpuid', default="", type=str,
                       help="Use CUDA on the listed devices.")



    opt = parser.parse_args()

    

    #generate hidden prepresentation
    
    if(opt.source_test_data != ""):
        print("Generating hidden representation for test data:",opt.source_test_data)
        if(opt.model == ""):
            logging.error("No model given to generate the hidden represnation");
            exit(-1);
        if(opt.model_type == ""):
            logging.error("No model type given to generate the hidden represnation");
            exit(-1);
        elif(opt.model_type == "OpenNMT"):
            data = toolkits.ONMT.generate(opt.source_test_data,opt.model,opt.representation,opt.gpuid)
        elif (opt.model_type == "vivisectONMT"):
            data = toolkits.vivisectONMT.generate(opt.source_test_data, opt.target_test_data, opt.model, opt.representation, opt.gpuid)
        elif (opt.model_type == "vivisectSockeye"):
            data = toolkits.vivisectSockeye.generate(opt.source_test_data, opt.target_test_data, opt.model,
                                                  opt.representation, opt.gpuid)
        else:
            logging.error("Unknown model type:",opt.model_type)
            exit(-1)
    elif(opt.hidden_representation != ""):
        with open(opt.hidden_representation) as f:
            data = json.load(f)
    else:
        logging.error("Neither testdata with model nor hidden representation given")
        exit(-1)

    #print("Number of sentences:",len(data.sentences))
    #for i in range(len(data.sentences)):
    #    print(data.sentences[i].data.shape)

    #annote hidden representation with labels
    if(opt.prediction_task == "BPE"):
        annotator.BPE.annotate(data);
    elif(opt.prediction_task == "SentenceLabels"):
        annotator.SentenceLabels.annotate(data,opt.label_file)
    elif(opt.prediction_task == "TokenLabels"):
        annotator.TokenLabels.annotate(data,opt.label_file)

    ## save hidden representation if necessry
    if(opt.hidden_representation_out != ""):
        f = open(opt.hidden_representation_out,'w')
        for i in range(len(data.sentences)):
            for j in range(len(data.sentences[i].words)):
                print("S"+str(i)+"T"+str(j),",".join([str(f) for f in data.sentences[i].data[j].tolist()]),file=f)
        f.close()
        

    #inpsect hidden representation
    if(opt.ml_technique  == "intrinsic"):
        result = inspector.Intrinsic.inspect(data)
    elif(opt.ml_technique == "classifier"):
        result = inspector.Classifier.inspect(data,opt.inspection_model,
                                              opt.load_inspection_model,
                                              opt.store_inspection_model,
                                              opt.inspection_model_input,
                                              opt.output_predictions);


    #store result if necessay

    #show result
        
    
if __name__ == '__main__':
    main()

              
