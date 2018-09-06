# Autoencoder

Train auto-encoder
```
python $NMTInspector/Inspect.py -ml_technique autoencoder 
                                -representation $rep 
                                -source_test_data $source_file
                                -target_test_data $target_file 
                                -model $model 
                                -model_type vivisectONMT 
                                -gpuid $gpu 
                                -inspection_model $classifier 
                                -store_inspection_model 
                                -inspection_model_input word 
                                -inspection_model_param $hiddenSize 
                                -output_predictions 
```

with
  * rep: Which representation to inspect. Possible values: EncoderWordEmbeddings,EncoderHiddenLayer,ContextVector,DecoderWordEmbeddings,DecoderHiddenLayer
  * source_file: Text file containing the source text
  * target_file: Text file containing the target text
  * model: NMT Model used for generating the representations
  * gpu: GPU used
  * classifier: where to store the classifier
  * hiddenSize: Hidden Size of the auto-encoder
  
Test auto-encoder
```
python $NMTInspector/Inspect.py -ml_technique autoencoder 
                                -representation $rep 
                                -source_test_data $source_file
                                -target_test_data $target_file 
                                -model $model 
                                -model_type vivisectONMT 
                                -gpuid $gpu 
                                -inspection_model $classifier 
                                -load_inspection_model 
                                -inspection_model_input word 
                                -inspection_model_param $hiddenSize 
                                -output_predictions 
```
with
  * rep: Which representation to inspect. Possible values: EncoderWordEmbeddings,EncoderHiddenLayer,ContextVector,DecoderWordEmbeddings,DecoderHiddenLayer
  * source_file: Text file containing the source text
  * target_file: Text file containing the target text
  * model: NMT Model used for generating the representations
  * gpu: GPU used
  * classifier: where to store the classifier
  * hiddenSize: Hidden Size of the auto-encoder
