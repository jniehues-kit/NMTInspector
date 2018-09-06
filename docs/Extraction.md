# Extraction

Extract feature to csv format
```
python $NMTInspector/Inspect.py -ml_technique none 
                                -representation $rep 
                                -source_test_data $source_file
                                -target_test_data $target_file
                                -model $model 
                                -model_type vivisectONMT 
                                -gpuid $gpu 
                                -hidden_representation_out $outfile
```

with
  * rep: Which representation to inspect. Possible values: EncoderWordEmbeddings,EncoderHiddenLayer,ContextVector,DecoderWordEmbeddings,DecoderHiddenLayer
  * source_file: Text file containing the source text
  * target_file: Text file containing the target text
  * model: NMT Model used for generating the representations
  * gpu: GPU used
  * outputfile: File to store the representations

