"""
"""

from evaluation import ConfusionMatrix

import random

class Perceptron(object):
    """
    """
    
    def __init__(self,):
        self.weights = None
    
    def fit(self, train, dev, learning_rate, nepochs, minff=5, maxff=float("+inf")):
        """
        Input: list of sentences, list of sentences, ...
        
        Estimates new weights from train data.
        Reports evaulation on dev data each epoch.
        Returns list of train and dev metrics over the epochs.
        """
        
        ## Init new weights
        self.classes = tuple(set([t.gold_label for t in train]))
        self.weights = dict()
        counts = dict()
        
        ## Iterate for each epoch
        for e in range(nepochs):
            ## Randomize order of exaples
            random.shuffle(train)
            ## Iterate each example
            for i,example in enumerate(train):
                # predict output scores (with current weights)
                _,pred = self.scores(example.features)
                # calculate update for each label
                for ptag, pscore in pred.items():
                    # calculate update direction
                    if ptag==example.gold_label:
                        d = 1-pscore
                    else:
                        d = 0-pscore
                    # calculate update for each feature
                    if ptag not in self.weights:
                        self.weights[ptag] = dict()
                    for f in example.features:
                        if f not in self.weights:
                            self.weights[ptag][f] = 0
                            counts[f] = 0
                        self.weights[ptag][f] += d*learning_rate
                        counts[f] += 1
                if i%10000==0:
                    print("Iteration: {:6d}    #seenFeatures {}".format(i, len(counts)))
            # Evaluate on dev
            pred_tokens = [self.scores(token.features)[0] for token in train]
            gold_tokens = [token.gold_label for token in train]
            ev = ConfusionMatrix.from_data(gold_tokens, pred_tokens)
            macro = ev.macro_f1()
            micro = ev.micro_f1()
            print("Epoch: {:3d}   TRAIN   micro: {} macro: {}".format(e, micro, macro))
            # Evaluate on train
            pred_tokens = [self.scores(token.features)[0] for token in dev]
            gold_tokens = [token.gold_label for token in dev]
            ev = ConfusionMatrix.from_data(gold_tokens, pred_tokens)
            macro = ev.macro_f1()["F1"]
            micro = ev.micro_f1()["F1"]
            print("Epoch: {:3d}     DEV   micro: {} macro: {}".format(e, micro, macro))
        ## Delete features seen to often/rarely during training
        for f in list(counts.keys()):
            if (counts[f]<minff) or (counts[f]>maxff):
                for t in self.weigthts:
                    del self.weights[t][f]
        ##
    
    def scores(self, feat_vec):
        """
        Takes a feature vector.
        And returns a score for each class and the predicted tag
        """
        scores = {t:0 for t in self.classes}
        for feat in feat_vec:
            for tag in self.weights:
                if feat in self.weights[tag]: # ingore unknown featuers
                    scores[tag] += self.weights[tag][feat]
        max_tag,_ = max(scores.items(), key=lambda x:x[1])
        return max_tag,scores


