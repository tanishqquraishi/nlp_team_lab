from data import LoadOntoNotes
from perceptron import Perceptron

## Load data
train_data = LoadOntoNotes("./train.col").get_sentences()
dev_data   = LoadOntoNotes("./dev.col").get_sentences()


## Extract features on sentence level
for s in train_data:
    s.extract_features()
for s in dev_data:
    s.extract_features()
    
## Print some example sentence
print()
for t in train_data[0].tokens:
    print("{:10s} ({:4s}): {}".format(t.text, t.gold_label, t.features))
print()


## Get single tokens
train = [token for sent in train_data for token in sent.tokens]
dev   = [token for sent in dev_data   for token in sent.tokens]


## Train model (TBD)
model = Perceptron()
train_hist = model.fit(train, dev, learning_rate=1, nepochs=5, lr_decay=0.01)
#model.save(".../.../") (TBD)

print("FINAL EVAL ON DEV DATA")
print(model._evaluate(dev))

## For model selection and testing
#model.predict_data(dev) (TBD)
#evaluation = ConfusionMatrix.from_sentence(dev_data)
#macro_f1 = evaluation.macro_f1()
#micro_f1 = evaluation.macro_f1()
#scores = evaluation.f1_scores()
#print("MacroF1: {:6.2f}  MicroF1: {:6.2f}".format(macro_f1, micro_f1))
#print(scores)
#
