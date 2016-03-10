from __future__ import division

import plac
from pathlib import Path
import random
from collections import Counter
import numpy as np
from itertools import cycle

import spacy.en
from spacy.strings import StringStore, hash_string

from thinc.linear.avgtron import AveragedPerceptron
from thinc.extra.eg import Example


class KerasModel(object):
    def __init__(self, widths, vocab_size=5000):
        from keras.models import Sequential
        from keras.layers import Embedding, Dense, TimeDistributedMerge
        from keras.layers.advanced_activations import ELU
        from keras.preprocessing.sequence import pad_sequences
        from keras.optimizers import SGD
        self.n_classes = widths[-1]
        self.vocab_size = vocab_size
        self.word_to_int = {}
        self.int_to_word = np.ndarray(shape=(vocab_size+1,), dtype='int64')
        self.model = Sequential()
        self.model.add(Embedding(vocab_size, widths[0]))
        self.model.add(TimeDistributedMerge(mode='ave'))
        for width in widths[1:-1]:
            layer = Dense(output_dim=hidden_width, init='he_normal', activation=ELU(1.0))
            self.model.add(layer)
        self.model.add(
            Dense(
                n_classes,
                init='zero',
                activation='softmax'))
        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy', optimizer=sgd)

    def train(self, examples, n_iter=5):
        counts = Counter(t.orth 
                    for doc, label in examples 
                        for t in doc)
        # 0 is referred for UNK
        for i, (word, count) in enumerate(counts.most_common(self.vocab_size-1)):
            self.word_to_int[word] = i+1
            self.int_to_word[i+1] = word

        X = pad_sequences([self.extract(doc) for doc, label in examples])
        y = np.zeros(shape=(len(examples), self.n_classes), dtype='int32')
        for i, (doc, label) in enumerate(examples):
            y[i, label] = 1
        return self.model.fit(X, y, validation_split=0.1,
                              show_accuracy=True, nb_epoch=n_iter)

    def extract(self, doc):
        x = np.ndarray(shape=(len(doc),), dtype='int32')
        for i, word in enumerate(doc):
            x[i] = self.word_to_int.get(word.orth, 0)
        return x

    def evaluate(self, examples):
        X = pad_sequences([self.extract(doc) for doc, label in examples])
        y = np.zeros(shape=(len(examples), self.n_classes), dtype='int32')
        for i, (doc, label) in enumerate(examples):
            y[i, label] = 1
        return self.model.evaluate(X, y, verbose=1)


class ThincModel(object):
    def __init__(self, nr_class):
        self.nr_class = nr_class
        self._eg = Example(nr_class=nr_class)
        self._model = AveragedPerceptron([])

    def Eg(self, doc, label=None):
        eg = self._eg
        eg.reset()

        features = []
        word_types = set()
        i = 0
        for token in doc[:-1]:
            next_token = doc[i+1]

            strings = (token.lower_, next_token.lower_)
            key = hash_string('%s_%s' % strings)
            feat_slot = 0
            feat_value = 1
            features.append(
                (0, token.lower, 1))
            features.append(
                (feat_slot, key, feat_value))
            i += 1

        eg.features = features
        if label is not None:
            eg.costs = [clas != label for clas in range(self.nr_class)]
        return eg

    def predict(self, doc):
        return self._model.predict_example(self.Eg(doc))

    def train(self, examples, n_iter=5):
        for i in range(n_iter):
            loss = 0
            random.shuffle(examples)
            negation_count = 0
            for doc, label in examples:
                eg = self.Eg(doc, label)
                self._model.train_example(eg)
                loss += eg.guess != label
                if any(t.orth_ == 'not' for t in doc):
                    negation_count += 1

            print(loss)
        self._model.end_training()
        print("%d documents had negation" % negation_count)

    def evaluate(self, examples):
        total = 0
        correct = 0
        for i, (doc, label) in enumerate(examples):
            eg = self.predict(doc)
            correct += eg.guess == label
            total += 1
        return correct / total

    def dump(self, loc):
        self._model.dump(loc)

    def load(self, loc):
        self._model.load(loc)


def read_data(nlp, data_dir, classes=('pos', 'neg')):
    for class_id, class_name in enumerate(classes):
        for i, filename in enumerate((data_dir / class_name).iterdir()):
            text = filename.open(encoding='utf8').read()
            doc = nlp(text)
            if len(doc) >= 1:
                yield doc, class_id


@plac.annotations(
    corpus_dir=("Corpus directory", "positional", None, Path),
    n_iter=("Number of iterations (epochs)", "option", "i", int),
    use_keras=("Use Keras model (not averaged perceptron", "flag", "k", bool),
)
def main(corpus_dir, n_iter=4, use_keras=False):
    print("Loading spaCy")
    nlp = spacy.en.English(parser=False, tagger=True, entity=False)
    print("Processing training data")
    train_data = list(read_data(nlp, corpus_dir / 'train'))
    if use_keras:
        model = KerasModel((50, 20, 2))
    else:
        model = ThincModel(2)
    print("Train", len(train_data))
    model.train(train_data, n_iter=n_iter)
    print("Evaluating")
    eval_data = list(read_data(nlp, corpus_dir / 'test'))
    print(model.evaluate(eval_data))
 

if __name__ == '__main__':
    plac.call(main)
