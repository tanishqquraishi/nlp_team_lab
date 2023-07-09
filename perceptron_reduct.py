"""
___date__: 30 / 04 / 2023
__author__: Florian Omiecienski (code), Tanishq Quraishi (docstrings, comments)

"""

from evaluation import ConfusionMatrix

import random
import numpy as np

class PerceptronReduct(object):
    """
    """
    
    def __init__(self,):
        self.weights = None
    
    def _feature_class_distribution(self, tokens):
        counts = dict()
        for t in tokens:
            c = t.gold_label
            for f in t.features:
                try:
                    tmp = counts[f]
                except KeyError:
                    counts[f] = tmp = dict()
                try:
                    tmp[c] += 1.0
                except KeyError:
                    tmp[c] = 1.0
        return counts
    
    def _count_feautres(self, tokens):
        """
        Counts each feature from a list of tokens.
        Returns dictionary with features as keys and counts as values.
        """
        counts = dict()
        for t in tokens:
            for f in t.features:
                if f not in counts:
                    counts[f] = 0
                counts[f] += 1
        return counts
    
    def _evaluate(self, tokens):
        """
        Given a list of tokens, do an evaluation
        """
        pred_tokens = [self.predict(token.features)[0] for token in tokens]
        gold_tokens = [token.gold_label for token in tokens]
        ev = ConfusionMatrix.from_data(gold_tokens, pred_tokens, nan=0)
        macro = ev.macro_f1()
        micro = ev.micro_f1()
        expanded = ev.f1_scores()
        return macro, micro, expanded
    
    def fit(self, src_data, tgt_data, learning_rate, nepochs, lr_decay=0.0, minff=5, maxff=float("+inf")):
        """
        Input: list of sentences, list of sentences, ...
        Parameters: src_data, tgt_data, learning rate, number of epochs,
                    learning rate decay (iteratively reduces LR after each epoch),
                    minimum feature frequency (features ocurring
                    less than 5 times ignored), maximum feature frequency. 
                    
        Estimates new weights from train data of src and tgt data.
        Reports evaulation on dev data each epoch.
        Returns list of train and dev metrics over the epochs.
        """
        ##
        src_train,src_dev = src_data
        tgt_train,tgt_dev = tgt_data
        train = src_train+tgt_train
        dev = src_dev+tgt_dev
        ## Count all feature-class cooccurences in trains
        self.classes = tuple(set([t.gold_label for t in train]))
        ## Use all features from both domain
        self.features = tuple(set([f for t in train for f in t.features]))
        ## For all features that appear in both domains, calculate similarity of class distributions, as feature-weight
        dist_src = self._feature_class_distribution(src_train)
        dist_tgt = self._feature_class_distribution(tgt_train)
        ##
        class_intersection = set([t.gold_label for t in src_train]).intersection(set([t.gold_label for t in tgt_train]))
        feature_weights = dict()
        for k in self.features:
            if (k not in dist_tgt):
                feature_weights[k] = 0.5
                continue
            if k not in dist_src:
                feature_weights[k] = 1.0
                continue
            a = np.array([dist_src[k][c] if c in dist_src[k] else 0.0 for c in class_intersection])
            b = np.array([dist_tgt[k][c] if c in dist_tgt[k] else 0.0for c in class_intersection])
            an = np.linalg.norm(a)
            bn = np.linalg.norm(b)
            if (an == 0) or (bn == 0):
                feature_weights[k] = 0.5
            else:
                feature_weights[k] = 0.5 + (np.dot(a,b)/(an*bn))
        

        ## Set features-weights that occure only in target domain to 1
        for f in self.features:
            try:
                tmp = feature_weights[f]
            except KeyError:
                feature_weights[f] = 1
        ## Init new weights (trainable parameters)
        self.weights = {c:{f:0 for f in self.features} for c in self.classes}
        print("Starting estimation for {} classes and {} features and {} train examples".format(len(self.classes), len(self.features), len(train)))
        
        ## Iterate for each epoch
        train_history = []
        for e in range(nepochs):
            ## Randomize order of exaples
            random.shuffle(train)
            ## Iterate each example
            for i,example in enumerate(train, start=1):
                #
                pred_tag,_ = self.predict(example.features)
                gold_tag = example.gold_label
                # If prediction is correct, do nothing
                if pred_tag==gold_tag:
                    continue
                for f in example.features:
                    # Else, increase weights at feature positions for gold
                    try: # try is faster than if check for keys in dict
                        self.weights[gold_tag][f] += learning_rate*feature_weights[f] ## weight by feature
                    except KeyError:
                        pass
                    # and decrease weights at feature positions for prediction
                    try:
                        self.weights[pred_tag][f] -= learning_rate*feature_weights[f] ## weight by feature
                    except KeyError:
                        pass
            # update learning rate
            learning_rate *= (1.0-lr_decay)
            print("LEARNING RATE is now", learning_rate)
            # Evaluate
            #macro, micro, _ = self._evaluate(train)
            #print("Epoch: {:<3d}       TRAIN   micro: {:6.2f} macro: {:6.2f}".format(e, micro["F1"], macro["F1"]))
            #train_eval = {"microF1":micro["F1"], "macroF1":macro["F1"]}
            train_eval = None
            #
            macro, micro, _ = self._evaluate(src_dev)
            print("Epoch: {:<3d}     DEV-SRC   micro: {:6.2f} macro: {:6.2f}".format(e, micro["F1"], macro["F1"]))
            #
            macro, micro, _ = self._evaluate(tgt_dev)
            print("Epoch: {:<3d}     DEV-TGT   micro: {:6.2f} macro: {:6.2f}".format(e, micro["F1"], macro["F1"]))
            #
            macro, micro, _ = self._evaluate(src_dev+tgt_dev)
            print("Epoch: {:<3d}     DEV-ALL   micro: {:6.2f} macro: {:6.2f}".format(e, micro["F1"], macro["F1"]))
            dev_eval = {"microF1":micro["F1"], "macroF1":macro["F1"]}
            #
            train_history.append({"train":train_eval, "dev":dev_eval})
        return train_history
    
    def predict(self, feat_vec):
        """
        Takes a feature vector.
        And returns the predicted tag and a score for each class
        """
        scores = {t:0 for t in self.classes}
        for feat in feat_vec:
            for tag in self.weights:
                try: # try is faster than if check for keys in dict
                    scores[tag] += self.weights[tag][feat]
                except KeyError: # ingore unknown featuers
                    pass
        max_tag,_ = max(scores.items(), key=lambda x:x[1])
        return max_tag,scores
    
    def predict_sentences(self, data):
        """
        Given a list of sentence objects.
        Preict each token and store result in respective token object.
        """
        for sent in data:
            for token in sent.tokens:
                ptag,_ = self.predict(token.features)
                token.pred_label = ptag
        return sent

