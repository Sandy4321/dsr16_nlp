Data Science Retreat 2016: Text Classification in Python
========================================================

Set up the code and download the data:

```
virtualenv env
source env/bin/activate
pip install -r requirements.txt
./download_data.sh
```

We'll start by doing sentiment analysis using spaCy and Keras:

* www.keras.io : The most popular deep learning library (in any language)
* www.spacy.io : Our NLP library

Further tasks:

* Classify questions, according to their answer type

* Visual Question Answering: https://avisingh599.github.io/deeplearning/visual-qa/

* Train a joke-making bot on the Reddit data set: https://archive.org/details/2015_reddit_comments_corpus

Use the GloVe word vectors with spaCy
-------------------------------------

```
USE_BLAS=1 pip install https://github.com/spacy-io/sense2vec/archive/master.zip
sputnik --name spacy install en_glove_cc_300_1m_vectors
```

