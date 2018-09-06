# NMTInspector

This tools should inspect hidden representation of NMT systems. Currently, it  supports to classify the hidden representation, to dump them and to train an auto-encoder on them. Furthermore, intrinsict properties can be calculated.

Currently, NMT models from OpenNMT-py are supported.


##Installation

* Python 3 is needed
* Install visisect
  * git clone https://github.com/jniehues-kit/vivisect.git
  * cd vivisect
  * pip install .
* Download NMTInspect
  * git clone https://github.com/jniehues-kit/NMTInspector.git
* Install OpenNMT-py
  * git clone https://github.com/OpenNMT/OpenNMT-py.git
  * Add openNMT to your python path export PYTHONPATH=$DIR/OpenNMT-py
* Install deepdish
  * pip install deepdish
* conda install scikit-learn
 
## Paramter


## Example use cases

### Sentence-level classification

The idea of this use case is to inspect if some property of a sentence is represented.
Therefore, it is possible to train a classifier, which uses a representation of the sentence as input.
We obtain the representation by summing over the representation of each token in the sentence [Example](docs/SentenceClassifier.md)

### Token-level classification

The idea of this use case is to inspect if some property of a token is represented.
Therefore, it is possible to train a classifier, which uses a representation of each token as input.
For EncoderWordEmbeddings and EncoderHiddenLayer the number of tokens equals the number of source tokens.
For the other representations the number of tokens represents the number of words +1 for each sentences, since als an
end-of-sentence token is predicted. [Example](docs/TokenClassifier.md)

### Extraction of hidden representation

Store hidden representation for external analysis [Example](docs/Extraction.md)

### Outlier detection
The idea is to detect hidden states that are not typical for the NMT system.
This is done by training a auto-encoder on the hidden state. In a second step the reconstruction error is measured.
Then the reconstruction error on unusual states should be lower than the one on un-unusual states. [Example](docs/Autoencoder.md)
