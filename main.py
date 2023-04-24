from data import LoadOntoNotes


## Load data
train = LoadOntoNotes("./train.col").get_sentences()
dev   = LoadOntoNotes("./dev.col").get_sentences()
print("'{}' ({}): {}".format(train[0].tokens[0].text, train[0].tokens[0].gold_label, train[0].tokens[0].features))


## Extract features on sentence level
for s in train:
    s.extract_features()
for s in dev:
    s.extract_features()
print("'{}' ({}): {}".format(train[0].tokens[0].text, train[0].tokens[0].gold_label, train[0].tokens[0].features))


## Get single tokens
train = [token for sent in train for token in sent.tokens]
dev   = [token for sent in dev   for token in sent.tokens]
print("'{}' ({}): {}".format(train[0].text, train[0].gold_label, train[0].features))


## Train model (TBD)
model = Perceptron(nclasses=None)
model.fit(train, dev)
model.save(".../.../")

