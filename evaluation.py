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
        
        return {"P":P, "R":R, "F1":F1}
    
    @staticmethod
    def from_data(gold, pred):
        """
        Create a new confusion matrix from two lists of labels.
        """
        # Count confusions and collect all used tags
        conf = dict()
        all_tags = set()
        for g,p in zip(gold,pred):
            if g not in conf:
                conf[g] = dict()
            if p not in conf[g]:
                conf[g][p] = 0
            conf[g][p] += 1
            all_tags.update([g,p])
        # Add missing zero-entries 
        # (eg. if a tag occures only in gold or in only prediction)
        for t in all_tags:
            if t not in conf:
                conf[t] = {k:0 for k in all_tags}
            else:
                for k in all_tags:
                    if k not in conf[t]:
                        conf[t][k] = 0
        ##
        return ConfusionMatrix(conf, all_tags)
    
    def get_confusions(self, tag):
        """
        Returns how often 'tag' was confused with another tag.
        """
        # copy dict !!!
        confs = dict(self.confusions[tag])
        return confs
    
    def errors(self, tag):
        """
        Returns the number of errors for 'tag'.
        """
        TP = self.confusions[tag][tag]
        FP = sum([self.confusions[tag][k] for k in self.all_tags if k!=tag])
        FN = sum([self.confusions[k][tag] for k in self.all_tags if k!=tag])
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
