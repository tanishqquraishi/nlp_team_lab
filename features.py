
class featureExt:
    """
    A class for extracting features from tokens in a sentence.
    """
    def __init__(self, token, sent):
        self.token = token 
        self.sent =  sent
    
    def prefix(self, n):
        if len(self.token.text)>=n:
            return "prefix="+self.token.text[:n]
        return None
    
    def suffix(self, n):
        if len(self.token.text)>=n:
            return "suffix="+self.token.text[-n:]
        return None
    
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
        Checks if characters in a token are digits.
        # Updated methods includes, for eg: "2,500".
        """
        for char in self.token.text:
            if char.isdigit():
                return True
            return False 


    def isPunct(self):
        """
        Checks if token is a punctuation. Excludes hyphen.
        """
        punct = [".", ",", "!", "?", "'", ";", "...", ":", "/"]
        return self.token.text in punct

    def isCapitalized(self): ## maybe rename this function, eg. typos
                                ## resolved
        """
        Checks if token is capitalized and excludes first token in the sentence.        
        """
        return self.token.text[0].isupper() and not self.isFirst()

    def case(self):
        features = {}
        features['is_all_uppercase'] = self.token.text.isupper()
        features['is_all_lowercase'] = self.token.text.islower()
        features['is_titlecase'] = self.token.text.istitle()
        features['is_other'] = not (self.token.text.isupper() or self.token.text.islower() or self.token.text.istitle())
        return features
    
    def isNNP(self):
    """
    Checks if the token and the next one's first character are each capitalized.
    Eg: Los Angeles.
    """
        index = self.sent.tokens.index(self.token)
        if index + 1 < len(self.sent.tokens):
            next_token = self.sent.tokens[index + 1]
            return self.token.text[0].isupper() and next_token.text[0].isupper()
       return None
