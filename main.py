from data import LoadOntoNotes
from perceptron import Perceptron

## Load data
train_data = LoadOntoNotes("./train.col").get_sentences()
dev_data   = LoadOntoNotes("./dev.col").get_sentences()
test_data  = LoadOntoNotes("./test.col").get_sentences()


## Extract features on sentence level
for s in train_data:
    s.extract_features()
for s in dev_data:
    s.extract_features()
for s in test_data:
    s.extract_features()


## Print some example sentence
print()
for t in train_data[0].tokens:
    print("{:10s} ({:4s}): {}".format(t.text, t.gold_label, t.features))
print()


## Get single tokens
train = [token for sent in train_data for token in sent.tokens]
dev   = [token for sent in dev_data   for token in sent.tokens]


## Train model
model = Perceptron()
train_hist = model.fit(train, dev, learning_rate=1, nepochs=1, lr_decay=0.01)


## Evaluate on dev data
print("FINAL EVAL ON TEST DATA")
ev = model.evaluate(test_data)
print("MacroF1: {:6.2f}  MicroF1: {:6.2f}".format(ev.macro_f1()["F1"], ev.micro_f1()["F1"]))
print(ev.f1_scores())
print()
ev.print()
#
