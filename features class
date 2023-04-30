
class featureExt:
    """
    A class for extracting features from tokens in a sentence.
    """
    def __init__(self, token, sent):
        self.token = token 
        self.sent =  sent

    def isFirst(self):
        """
        Checks if it's the first token in the sentence. 
        """
        sent_list = self.sent.split()
        return sent_list[0] == self.token

    def isLast(self):
        """
        Checks if it's the last token in the sentence, excludes punct.
        """
        punct = [".", "!", "?", "..."]
        sent_list = self.sent.split()
        return self.token not in punct and sent_list[-1] == self.token

    def isDigit(self):
        """
        Checks if token is a digit/ CD (cardinal number).
        # Note: CD includes "2,500" or "million" too?
        """
        return self.token.isdigit()

    def isPunct(self):
        """
        Checks if token is a punctuation. Excludes hyphen.
        """
        punct = [".", ",", "!", "?", "'", ";", "...", ":", "/"]
        return self.token in punct

    def isNNP(self):
        """
        Checks if token is proper noun or not. Excludes first token in the sent.
        # Note: what if the first token is an NNP? 
        """
        return self.token[0].isupper() and not self.isFirst()

    def case(self):
        features = {}
        features['is_all_uppercase'] = self.token.isupper()
        features['is_all_lowercase'] = self.token.islower()
        features['is_titlecase'] = self.token.istitle()
        features['is_other'] = not (self.token.isupper() or self.token.islower() or self.token.istitle())
        return features
