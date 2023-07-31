"""
___date__: 23 / 04 / 2023
__author__: Tanishq Quraishi and Florian Omiecienski

"""
from features import featureExt

from datasets import load_dataset


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

    def copy(self, name):
        for tokens in self.tokens:
            #features = []
            featureExt(None, None).copy_features(tokens, name) #.append(features)
        return tokens  
    
    def extract_features(self):
        """
        Extracts features for each token in this sentence.
        """
        for token in self.tokens:
            ext = featureExt(token, self)
            features = []

            # current,prev. and next token
            features.append(ext.tokenText(1))
            features.append(ext.tokenText(-1))
            

            # suffix features
            s = ext.suffix(1)       ## New: Like last character of token, maybe helpfull for plural Nouns
            if s is not None:
                features.append(s)
            s = ext.suffix(2)
            if s is not None:
                features.append(s)
            s = ext.suffix(3)
            if s is not None:
                features.append(s)

            
            # prefix

            s = ext.prefix(1)
            if s is not None:
                features.append(s)
            s = ext.prefix(2)
            if s is not None:
                features.append(s)
            s = ext.prefix(3)
            if s is not None:
                features.append(s)

            #

            if ext.isFirst():
                features.append("isFirst")
           
            if ext.isLast():
                features.append("isLast")
           
            #
            if ext.isDigit():
                features.append("isDigit")
            
            
            #
            if ext.isPunct():
                features.append("isPunct")
                features.append("punctType="+token.text)       ## New: Gives the Punctuation itself, helpfull for '', or brackets
            
            #
            if ext.isCapitalized():
                features.append("isCapitalized")
            #
            for f,v in ext.case().items():
                if v is True:
                    features.append(f)

            #
            if ext.isNNP():
                features.append("isNNP")
            token.features = features

           

    
    def str(self):
        text = [t.text for t in self.tokens]
        gold = [t.gold_label for t in self.tokens]
        pred = [str(t.pred_label) for t in self.tokens]
        width = [max(len(text[i]),len(gold[i]),len(pred[i])) for i in range(len(self.tokens))]
        
        o1,o2,o3 = "","",""
        for t,g,p,w in zip(text, gold, pred, width):
            o1 += ("  {:"+str(w)+"s}").format(t)
            o2 += ("  {:"+str(w)+"s}").format(g)
            o3 += ("  {:"+str(w)+"s}").format(p)
        m = max(len(o1),len(o2),len(o3))
        o =  ("-"*m+"\n"+o1+"\n"+"-"*m+"\n"+o2+"\n"+"-"*m+"\n"+o3+"\n"+"-"*m+"\n")
        return o


def load_TwitterPos():
    """
    Returns train,dev and test split of the Ritter Dataset.
    For details see  https://huggingface.co/datasets/strombergnlp/twitter_pos
    """
    ##
    data = load_dataset("strombergnlp/twitter_pos", "ritter")
    train = data["train"]
    dev = data["validation"]
    test = data["test"]
    pos_tags = train.features["pos_tags"].feature.names
    print(pos_tags)
    ##
    train = [Sentence([Token(text=t, gold_label=pos_tags[l]) for t,l in zip(s["tokens"], s["pos_tags"])]) for s in train]
    dev = [Sentence([Token(text=t, gold_label=pos_tags[l]) for t,l in zip(s["tokens"], s["pos_tags"])]) for s in dev]
    test = [Sentence([Token(text=t, gold_label=pos_tags[l]) for t,l in zip(s["tokens"], s["pos_tags"])]) for s in test]
    #
    return train, dev, test


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
