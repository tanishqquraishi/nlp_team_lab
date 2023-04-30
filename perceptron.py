"""
"""

from evaluation import ConfusionMatrix

import random

class Perceptron(object):
    """
    """
    
    def __init__(self,):
        self.weights = None
    
    def _count_feautres(self, tokens):
        counts = dict()
        for t in tokens:
            for f in t.features:
                if f not in counts:
                    counts[f] = 0
                counts[f] += 1
        return counts
    
    def fit(self, train, dev, learning_rate, nepochs, lr_decay=0.0, minff=5, maxff=float("+inf")):
        """
        Input: list of sentences, list of sentences, ...
        
        Estimates new weights from train data.
        Reports evaulation on dev data each epoch.
        Returns list of train and dev metrics over the epochs.
        """
        
        ## Count all classes in train (add UNK)
        self.classes = tuple(set([t.gold_label for t in train]))
        ## Count all features
        counts = self._count_feautres(train)
        # keep only features seen often but not too often (minff, maxff)
        for f in list(counts.keys()):
            if (counts[f]<minff) or (counts[f]>maxff):
                del counts[f]
        self.features = tuple(counts.keys())
        ## Init new weights
        self.weights = {c:{f:0 for f in self.features} for c in self.classes}
        
        print("Starting estimation for {} classes and {} features and {} train examples".format(len(self.classes), len(self.features), len(train)))
        
        ## Iterate for each epoch
        for e in range(nepochs):
            ## Randomize order of exaples
            random.shuffle(train)
            ## Iterate each example
            for i,example in enumerate(train):
                # predict output scores (with current weights)
                _,pred = self.scores(example.features)
                # calculate update for each class
                for ptag, pscore in pred.items():
                    # calculate update direction for this class
                    if ptag==example.gold_label:
                        d = 1-pscore
                    else:
                        d = 0-pscore
                    # update each feature for this class
                    for f in example.features:
                        if f in self.features:
                            self.weights[ptag][f] += d*learning_rate
                # update learning rate
                learning_rate *= (1-lr_decay)
                if i%20000==0:
                    print("Iteration: {:<7d}    learning-rate: {}".format(i, learning_rate))
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


