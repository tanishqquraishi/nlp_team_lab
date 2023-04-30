from data import LoadOntoNotes
from perceptron import Perceptron

## Load data
train_data = LoadOntoNotes("./train.col").get_sentences()
dev_data   = LoadOntoNotes("./dev.col").get_sentences()
print("'{}' ({}): {}".format(train_data[0].tokens[0].text, train_data[0].tokens[0].gold_label, train_data[0].tokens[0].features))


## Extract features on sentence level
for s in train_data:
    s.extract_features()
for s in dev_data:
    s.extract_features()
print("'{}' ({}): {}".format(train_data[0].tokens[0].text, train_data[0].tokens[0].gold_label, train_data[0].tokens[0].features))


## Get single tokens
train = [token for sent in train_data for token in sent.tokens]
dev   = [token for sent in dev_data   for token in sent.tokens]
print("'{}' ({}): {}".format(train[0].text, train[0].gold_label, train[0].features))


## Train model (TBD)
model = Perceptron()
model.fit(train, dev, learning_rate=1.0, nepochs=10)
#model.save(".../.../") (TBD)

## For model selection and testing
#model.predict_data(dev) (TBD)
#evaluation = ConfusionMatrix.from_sentence(dev_data)
#macro_f1 = evaluation.macro_f1()
#micro_f1 = evaluation.macro_f1()
#scores = evaluation.f1_scores()
#print("MacroF1: {:6.2f}  MicroF1: {:6.2f}".format(macro_f1, micro_f1))
#print(scores)
#