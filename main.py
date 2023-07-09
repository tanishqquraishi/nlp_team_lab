"""
___date__: 24 / 04 / 2023
__author__: Florian Omiecienski

"""

from data import LoadOntoNotes, load_TwitterPos
from perceptron import Perceptron
from perceptron_weight import PerceptronWeight
from perceptron_reduct import PerceptronReduct
from evaluation import ConfusionMatrix

from pprint import pprint


def evaluate(test, src_name, tgt_name):
    """
    Evaluated predicted data and print results.
    test: predicted data, list of sentence objects
    src/tgt_name: strings, specifing the setting of evaluation.
    """
    ev = ConfusionMatrix.from_sentences(test, nan=0)
    print(src_name,"->",tgt_name)
    print("MacroF1: {:6.2f}  MicroF1: {:6.2f}".format(ev.macro_f1()["F1"], ev.micro_f1()["F1"]))
    pprint(ev.f1_scores())
    print("\n\n")


## Load data
ontoNotes_train = LoadOntoNotes("./train.col").get_sentences()
ontoNotes_dev   = LoadOntoNotes("./dev.col").get_sentences()
ontoNotes_test  = LoadOntoNotes("./test.col").get_sentences()
twitter_train, twitter_dev, twitter_test = load_TwitterPos()

## Extract features on sentence level (same features for both datasets)
for s in ontoNotes_train+ontoNotes_dev+ontoNotes_test:
    s.extract_features()
for s in twitter_train+twitter_dev+twitter_test:
    s.extract_features()

## Print some example sentence
print()
for t in ontoNotes_train[0].tokens:
    print("{:10s} ({:4s}): {}".format(t.text, t.gold_label, t.features))
print()
for t in twitter_train[0].tokens:
    print("{:10s} ({:4s}): {}".format(t.text, t.gold_label, t.features))
print()

## Convert train data to list of tokens
train_onto = [token for sent in ontoNotes_train for token in sent.tokens]
dev_onto = [token for sent in ontoNotes_dev for token in sent.tokens]
train_twitter = [token for sent in twitter_train for token in sent.tokens]
dev_twitter = [token for sent in twitter_dev for token in sent.tokens]

### Train model (OntoNotes)
#p = Perceptron()
#p.fit(train_onto, dev_onto, learning_rate=1, nepochs=5, lr_decay=0.01)
#p.predict_sentences(ontoNotes_test)
#p.predict_sentences(twitter_test)
#evaluate(ontoNotes_test, "OntoNotes", "OntoNotes")
#evaluate(twitter_test, "OntoNotes", "Twitter")

### Train model (Twitter)
#p = Perceptron()
#p.fit(train_twitter, dev_twitter, learning_rate=1, nepochs=5, lr_decay=0.01)
#p.predict_sentences(ontoNotes_test)
#p.predict_sentences(twitter_test)
#evaluate(ontoNotes_test, "Twitter", "OntoNotes")
#evaluate(twitter_test, "Twitter", "Twitter")
#
### Train model (ALL)
#p = Perceptron()
#p.fit(train_twitter+train_onto, dev_twitter+dev_onto, learning_rate=1, nepochs=5, lr_decay=0.01)
#p.predict_sentences(ontoNotes_test)
#p.predict_sentences(twitter_test)
#evaluate(ontoNotes_test, "ALL", "OntoNotes")
#evaluate(twitter_test, "ALL",   "Twitter")

## Train model (ALL+Reduct)
p = PerceptronReduct()
p.fit(src_data=(train_onto, dev_onto), tgt_data=(train_twitter, dev_twitter), learning_rate=1, nepochs=5, lr_decay=0.01)
p.predict_sentences(ontoNotes_test)
p.predict_sentences(twitter_test)
evaluate(ontoNotes_test, "ALL+Reduct", "OntoNotes")
evaluate(twitter_test, "ALL+Reduct",   "Twitter")

## Train model (ALL+Weight)
p = PerceptronWeight()
p.fit(src_data=(train_onto, dev_onto), tgt_data=(train_twitter, dev_twitter), learning_rate=5, nepochs=5, lr_decay=0.01)
p.predict_sentences(ontoNotes_test)
p.predict_sentences(twitter_test)
evaluate(ontoNotes_test, "ALL+Weight", "OntoNotes")
evaluate(twitter_test, "ALL+Weight",   "Twitter")

## Train model (ALL+Frust)
# Create new domain-specific features
copy_features(ontoNotes_train, "ontoNotes")
copy_features(ontoNotes_dev, "ontoNotes")
copy_features(ontoNotes_test, "ontoNotes")
copy_features(twitter_train, "twitter")
copy_features(twitter_dev, "twitter")
copy_features(twitter_test, "twitter")
# Train default perceptron on these features
p = Perceptron()
p.fit(train_twitter+train_onto, dev_twitter+dev_onto, learning_rate=1, nepochs=5, lr_decay=0.01)
p.predict_sentences(ontoNotes_test)
p.predict_sentences(twitter_test)
evaluate(ontoNotes_test, "ALL+Frust", "OntoNotes")
evaluate(twitter_test, "ALL+Frust",   "Twitter")
