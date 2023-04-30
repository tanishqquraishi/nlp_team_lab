
class featureExt:
    """
    A class for extracting features from tokens in a sentence.
    """
    def __init__(self, token, sent):
        self.token = token 
        self.sent =  sent
    
    def prefix(self, n):
        return "prefix="+self.token.text[:n]
    
    def suffix(self, n):
        return "suffix="+self.token.text[-n:]
    
    def isFirst(self):
        """
        Checks if it's the first token in the sentence. 
        """
        return self.sent.tokens[0].text == self.token.text

    def isLast(self):
        """
        Checks if it's the last token in the sentence, excludes punct.
        """
        punct = [".", "!", "?", "..."]
        return self.token.text not in punct and self.sent.tokens[-1].text == self.token.text

    def isDigit(self):
        """
        Checks if token is a digit/ CD (cardinal number).
        # Note: CD includes "2,500" or "million" too?
        """
        return self.token.text.isdigit()

    def isPunct(self):
        """
        Checks if token is a punctuation. Excludes hyphen.
        """
        punct = [".", ",", "!", "?", "'", ";", "...", ":", "/"]
        return self.token.text in punct

    def isNNP(self): ## maybe rename this function, eg. typos
        """
        Checks if token is proper noun or not. Excludes first token in the sent.
        # Note: what if the first token is an NNP? 
        """
        return self.token.text[0].isupper() and not self.isFirst()

    def case(self):
        features = {}
        features['is_all_uppercase'] = self.token.text.isupper()
        features['is_all_lowercase'] = self.token.text.islower()
        features['is_titlecase'] = self.token.text.istitle()
        features['is_other'] = not (self.token.text.isupper() or self.token.text.islower() or self.token.text.istitle())
        return features
