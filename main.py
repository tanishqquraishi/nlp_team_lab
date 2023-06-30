"""
___date__: 24 / 04 / 2023
__author__: Florian Omiecienski

"""

from data import LoadOntoNotes, load_TwitterPos
from perceptron import Perceptron
from evaluation import ConfusionMatrix

## Load data
ontoNotes_train = LoadOntoNotes("./train.col").get_sentences()
ontoNotes_dev = LoadOntoNotes("./dev.col").get_sentences()
ontoNotes_test = LoadOntoNotes("./test.col").get_sentences()

twitter_train, twitter_dev, twitter_test = load_TwitterPos()

## Extract features on sentence level
for s in ontoNotes_train+ontoNotes_dev+ontoNotes_test+twitter_train+twitter_dev+twitter_test:
    s.extract_features()


## Print some example sentence
print()
for t in ontoNotes_train[0].tokens:
    print("{:10s} ({:4s}): {}".format(t.text, t.gold_label, t.features))
print()
for t in twitter_train[0].tokens:
    print("{:10s} ({:4s}): {}".format(t.text, t.gold_label, t.features))
print()

## Train model (OntoNotes)
model = Perceptron()
train_hist = model.fit([token for sent in ontoNotes_train for token in sent.tokens], 
                       [token for sent in ontoNotes_dev   for token in sent.tokens],
                       learning_rate=1, nepochs=5, lr_decay=0.01)
#
p.predict_sentences(ontoNotes_test)
ev = ConfusionMatrix.from_sentences(ontoNotes_test, nan=0)
print("OntoNotes -> OntoNotes")
print("MacroF1: {:6.2f}  MicroF1: {:6.2f}".format(ev.macro_f1()["F1"], ev.micro_f1()["F1"]))
p.predict_sentences(twitter_test)
ev = ConfusionMatrix.from_sentences(twitter_test, nan=0)
print("OntoNotes -> Twitter")
print("MacroF1: {:6.2f}  MicroF1: {:6.2f}".format(ev.macro_f1()["F1"], ev.micro_f1()["F1"]))


# Train model (Twitter)
model = Perceptron()
train_hist = model.fit([token for sent in twitter_train for token in sent.tokens], 
                       [token for sent in twitter_dev   for token in sent.tokens],
                       learning_rate=1, nepochs=5, lr_decay=0.01)
#
p.predict_sentences(ontoNotes_test)
ev = ConfusionMatrix.from_sentences(ontoNotes_test, nan=0)
print("Twitter -> OntoNotes")
print("MacroF1: {:6.2f}  MicroF1: {:6.2f}".format(ev.macro_f1()["F1"], ev.micro_f1()["F1"]))
p.predict_sentences(twitter_test)
ev = ConfusionMatrix.from_sentences(twitter_test, nan=0)
print("Twitter -> Twitter")
print("MacroF1: {:6.2f}  MicroF1: {:6.2f}".format(ev.macro_f1()["F1"], ev.micro_f1()["F1"]))
