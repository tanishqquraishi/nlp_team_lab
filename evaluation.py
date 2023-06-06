class ConfusionMatrix(object):
    """
    An evaluation class.
    Can be used to see what kind of errors are made.
    """
    def __init__(self, confusions, all_tags, handle_nan=None):
        """
        Accepts a nested dictinary of confusiuons and a list of all labels.
        Set 'handle_nan' to control what happens if an evalution metric 
        is undefined (None: returns NaN, otherwise returns 'handle_nan').
        """
        self.confusions = confusions
        self.all_tags = all_tags
        self.nan = float("NaN") if handle_nan is None else handle_nan
    
    def errors2scores(self, errors):
        """
        Converts errors to evaluaton scores.
        """
        d = (errors["TP"] + errors["FP"])
        P = (errors["TP"] / d ) if d!= 0 else self.nan
        #
        d = (errors["TP"] + errors["FN"])
        R = (errors["TP"] / d) if d!= 0 else self.nan
        #
        d = (P+R)
        F1 = (2*P*R / d) if d!=0 else self.nan
        
        return {"P":P*100, "R":R*100, "F1":F1*100}
    
    @staticmethod
    def from_data(gold, pred, nan=None):
        """
        Create a new confusion matrix from two lists of labels.
        For what 'nan' is doing, please check the constructor.
        """
        # Check that both sequences are same length
        assert(len(gold)==len(pred))
        # Count confusions and collect all used tags
        conf = dict()
        all_tags = set()
        for g,p in zip(gold,pred):
            if g not in conf:
                conf[g] = dict()
            if p not in conf[g]:
                conf[g][p] = [0, None]
            conf[g][p][0] += 1
            all_tags.update([g,p])
        # Add missing zero-entries 
        # (eg. if a tag occures only in gold or in only prediction)
        for t in all_tags:
            if t not in conf:
                conf[t] = {k:[0, None] for k in all_tags}
            else:
                for k in all_tags:
                    if k not in conf[t]:
                        conf[t][k] = [0, None]
        ##
        return ConfusionMatrix(conf, all_tags, nan)
    
    @staticmethod
    def from_sentences(data, nan=None):
        """
        Create a new confusion matrix from a lists of predicted sentence objects.
        Also stores sentences where error occured.
        For what 'nan' is doing, please check the constructor.
        """
        # Count confusions and collect all used tags
        conf = dict()
        all_tags = set()
        for sent in data:
            gold = [token.gold_label for token in sent.tokens]
            pred = [token.pred_label for token in sent.tokens]
            for g,p in zip(gold,pred):
                if g not in conf:
                    conf[g] = dict()
                if p not in conf[g]:
                    conf[g][p] = [0, []]
                conf[g][p][0] += 1
                conf[g][p][1].append(sent) # store sentence where error occured
                all_tags.update([g,p])
        # Add missing zero-entries 
        # (eg. if a tag occures only in gold or in only prediction)
        for t in all_tags:
            if t not in conf:
                conf[t] = {k:[0, []] for k in all_tags}
            else:
                for k in all_tags:
                    if k not in conf[t]:
                        conf[t][k] = [0, []]
        ##
        return ConfusionMatrix(conf, all_tags, nan)
    
    def get_confusions(self, tag):
        """
        Returns how often 'tag' was confused with another tag.
        """
        # copy dict !!!
        confs = dict(self.confusions[tag])
        return confs
    
    def get_sentences(self, gold, pred):
        return list(self.confusions[gold][pred][1])
    
    def errors(self, tag):
        """
        Returns the number of errors for 'tag'.
        """
        TP = self.confusions[tag][tag][0]
        FP = sum([self.confusions[tag][k][0] for k in self.all_tags if k!=tag])
        FN = sum([self.confusions[k][tag][0] for k in self.all_tags if k!=tag])
        return {"TP":TP, "FP":FP, "FN":FN}
    
    def f1_scores(self):
        """
        Returns all scores for each label.
        """
        scores = dict()
        for tag in self.confusions:
            err = self.errors(tag)
            scores[tag] = self.errors2scores(err)
        return scores
    
    def micro_f1(self):
        """
        Returns a micro averaged scores.
        """
        # first collect errors for all tags
        err = {"TP":0, "FP":0, "FN":0}
        for tag in self.confusions:
            err_ = self.errors(tag)
            err["TP"] += err_["TP"]
            err["FP"] += err_["FP"]
            err["FN"] += err_["FN"]
        # calculate evalution scores from errors
        return self.errors2scores(err)
    
    def macro_f1(self):
        """
        Returns macro averaged scores.
        """
        # Calculte scores for each tag
        scores = {"P":0, "R":0, "F1":0}
        for tag in self.confusions:
            err = self.errors(tag)
            scores_ = self.errors2scores(err)
            scores["P"]  += scores_["P"] 
            scores["R"]  += scores_["R"] 
            scores["F1"] += scores_["F1"]
        # and average them later on
        scores["P"] /= len(self.confusions)
        scores["R"] /= len(self.confusions)
        scores["F1"] /= len(self.confusions)
        return scores
    
    def print(self, ignore_perfect=True):
        """
        Prints out the confusion matrix.
        To hide zero-rows set ignore_perfect=True
        """
        if ignore_perfect is False:
            ## Select all tags for printing
            all_tags = self.all_tags
        else:
            ## Select only tags with confusions for printing
            all_tags = []
            for t1 in self.all_tags:
                confs = self.get_confusions(t1)
                if any([c[0]!=0 for t2, c in confs.items() if t2!=t1]):
                    all_tags.append(t1)
        ## Do printing 
        print(" ".join(["      ",]+["{:^6s}".format(t) for t in self.all_tags]))
        for t1 in all_tags:
            confs = self.get_confusions(t1)
            print("{:>6s}".format(t1,), end=" ")
            for t2 in self.all_tags:
                print("{:^6d}".format(confs[t2][0]), end=" ")
            print()


if __name__ == "__main__":
    """
    Testing goes here
    """
    a = ["A", "B", "C", "A", "B", "C", "A", "B", "C", "E"]
    b = ["A", "A", "A", "B", "B", "B", "C", "C", "C", "C"]
    confmat = ConfusionMatrix.from_data(a, b)
    print("Confusions of 'A':")
    print(confmat.get_confusions('A'))
    print("Confusions of 'C':")
    print(confmat.get_confusions('C'))
    print("Errors on 'A':")
    print(confmat.errors('A'))
    print("MACRO:")
    print(confmat.macro_f1())
    print("MICRO:")
    print(confmat.micro_f1())
    print("No Avg.:")
    print(confmat.f1_scores())
    """
    Testing goes here
    """
    from data import LoadOntoNotes
    dl = LoadOntoNotes("./train.col")
    sentences = dl.get_sentences()
    #
    labels1 = [t.gold_label for t in sentences[0].tokens]
    labels2 = [t.gold_label for t in sentences[1].tokens]
    #
    confmat = ConfusionMatrix.from_data(labels1, labels1)
    print("\n\n\n",confmat.f1_scores(),"\n\n")
    #
    confmat = ConfusionMatrix.from_data(labels1, labels2)
