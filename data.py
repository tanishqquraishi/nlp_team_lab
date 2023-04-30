from features import featureExt



class Token(object):
    """
    A wrapper class holding token information.
    """
    def __init__(self, text, gold_label=None, pred_label=None, feats=None):
        self.text = text
        self.gold_label = gold_label
        self.pred_label = pred_label
        self.features = feats


class Sentence(object):
    """
    A sentence class holding tokens and for extacting features.
    """
    def __init__(self, tokens):
        self.tokens = tokens
    
    def extract_features(self,):
        """
        Extracts features for each token in this sentence.
        """
        for token in self.tokens:
            ext = featureExt(token, self)
            features = []
            # lexical? features
            s = ext.suffix(2)
            if s is not None:
                features.append(s)
            #
            s = ext.suffix(3)
            if s is not None:
                features.append(s)
            #
            s = ext.prefix(2)
            if s is not None:
                features.append(s)
            #
            s = ext.prefix(3)
            if s is not None:
                features.append(s)
            #
            if ext.isFirst():
                features.append("isFirst")
            if ext.isLast():
                features.append("isLast")
            if ext.isDigit():
                features.append("isDigit")
            if ext.isPunct():
                features.append("isPunct")
            if ext.isNNP():
                features.append("isNNP")
            for f,v in ext.case().items():
                if v is True:
                    features.append(f)
            token.features = features


class LoadOntoNotes:
    """
    Takes a path, reads the file and returns token and POS pairs 
    ...
    
    Attributes
    ----------
    
    path : str
    
    Methods
    -------
    read_file():
        opens file from path
        
    get_sentences():
        returns a list of sentences.
        a sentence is a list of pairs of words and their POS tags.
    """
    def __init__(self, path):
        self.path = path 
    
    def read_file(self):
        corpus = open(self.path)
        return corpus 
    
    def get_sentences(self):
        sentences = []
        tok_pos_pairs = []
        for lines in self.read_file():
            toks = lines.strip().split("\t")
            if len(toks) == 2:  
                tok_pos_pairs.append(Token(toks[0], toks[1]))
            else:
                sentences.append(Sentence(tok_pos_pairs))
                tok_pos_pairs = []
        if tok_pos_pairs:
            sentences.append(Sentence(tok_pos_pairs))
        return sentences


if __name__ == "__main__":
    """
    Testing goes here
    """
    lon = LoadOntoNotes("train.col")
    print(lon.get_sentences())
