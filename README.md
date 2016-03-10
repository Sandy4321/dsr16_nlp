Data Science Retreat 2016: Text Classification in Python
========================================================

Set up the code and download the data:

```
virtualenv env
source env/bin/activate
pip install -r requirements.txt
./download_data.sh
```

Use the GloVe word vectors with spaCy
-------------------------------------

```
USE_BLAS=1 pip install https://github.com/spacy-io/sense2vec/archive/master.zip
sputnik --name spacy install en_glove_cc_300_1m_vectors
```

