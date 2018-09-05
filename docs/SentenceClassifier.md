# Sentence Classifier

1. Train classifier:
```
python $NMTInspector/Inspect.py -prediction_task SentenceLabels 
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
                                -inspection_model_input sentence 
                                -output_predictions
```

with
  * source_file: Text file containing the source text
  * target_file: Text file containing the target text
  * label_file: File containing the labels for the classifier
  * model: NMT Model used for generating the representations
  * gpu: GPU used
  * classifier: where to store the classifier