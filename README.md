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