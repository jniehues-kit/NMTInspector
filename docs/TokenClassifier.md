# Token Classifier

## Train classifier:
```
python $NMTInspector/Inspect.py -prediction_task TokenLabels 
                                -ml_technique classifier 
                                -representation $rep 
                                -source_test_data $source_file
                                -target_test_data $target_file 
                                -label_file $label_file
                                -model $model 
                                -model_type vivisectONMT 
                                -gpuid $gpu 
                                -inspection_model $classifier 
                                -store_inspection_model 
                                -inspection_model_input word
                                -output_predictions
```

with
  * rep: Which representation to inspect. Possible values: EncoderWordEmbeddings,EncoderHiddenLayer,ContextVector,DecoderWordEmbeddings,DecoderHiddenLayer
  * source_file: Text file containing the source text
  * target_file: Text file containing the target text
  * label_file: File containing the labels for the classifier
  * model: NMT Model used for generating the representations
  * gpu: GPU used
  * classifier: where to store the classifier

## Use classifier on test data

```
python $NMTInspector/Inspect.py -prediction_task TokenLabels 
                                -ml_technique classifier 
                                -representation $rep 
                                -source_test_data $source_file
                                -target_test_data $target_file 
                                -label_file $label_file
                                -model $model 
                                -model_type vivisectONMT 
                                -gpuid $gpu 
                                -inspection_model $classifier 
                                -load_inspection_model 
                                -inspection_model_input word
                                -output_predictions
```

with
  * rep: Which representation to inspect. Possible values: EncoderWordEmbeddings,EncoderHiddenLayer,ContextVector,DecoderWordEmbeddings,DecoderHiddenLayer
  * source_file: Text file containing the source text
  * target_file: Text file containing the target text
  * label_file: File containing the labels for the classifier
  * model: NMT Model used for generating the representations
  * gpu: GPU used
  * classifier: where to store the classifier
