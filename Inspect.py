#!/usr/bin/env python


import sys

import toolkits.ONMT
import argparse
import logging
#import representation.DataSet
import annotator.BPE
import inspector.Intrinsic
import json



logging.basicConfig(format='%(message)s')


def main():

    parser = argparse.ArgumentParser(description='Inspect.py')
    
    
    parser.add_argument('-prediction_task', type=str,default="",
                        choices=['BPE','SentenceLabels'],
                        help="""Predicition task to be used to analyse the hidden representations""")

    parser.add_argument('-ml_technique', type=str, default="intrinsic",
                        choices=['intrinsic','supervised','unsupervised'],
                        help="""Technique used to analse the hidden representations""")


    parser.add_argument('-representation', type=str, default="EncoderWordEmbeddings",
                        choices=['EncoderWordEmbeddings','EncoderHiddenLayer'],
                        help="""EncoderWordEmbeddings""")
    
    parser.add_argument('-source_test_data', type=str, default="",
                        help="""Path to the input data""")

    parser.add_argument('-model', type=str, default="",
                        help="""Model that should be analyzed""")

    parser.add_argument('-model_type', type=str, default="",
                        choices=['OpenNMT'],
                        help="""Framework used to train the model""")


    parser.add_argument('-hidden_representation', type=str, default="",
                        help="""JSON File containing hidden representation from previous run""")

    parser.add_argument('-hidden_representation_out', type=str, default="",
                        help="""JSON File to store the hidden representation""")

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
        else:
            logging.error("Unknown model type:",opt.model_type)
            exit(-1)
    elif(opt.hidden_representation != ""):
        with open(opt.hidden_representation) as f:
            data = json.load(f)
    else:
        logging.error("Neither testdata with model nor hidden representation given")
        exit(-1)

    #annote hidden representation with labels
    if(opt.prediction_task == "BPE"):
        annotator.BPE.annotate(data);
    elif(opt.prediction_task == "SentenceLabels"):
        annotator.SentenceLabels.annotate(data,opt.label_file)

    ## save hidden representation if necessry
    #if(opt.hidden_representation_out != ""):
        

    #inpsect hidden representation
    if(opt.ml_technique  == "intrinsic"):
        result = inspector.Intrinsic.inspect(data)


    #store result if necessay

    #show result
        
    
if __name__ == '__main__':
    main()

              
