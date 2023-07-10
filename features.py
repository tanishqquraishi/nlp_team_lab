"""
___date__: 30 / 04 / 2023
__author__: Tanishq Quraishi and Florian Omiecienski

"""
class featureExt:
    """
    A class for extracting features from tokens in a sentence.
    """
    def __init__(self, token, sent):
        self.token = token 
        self.sent =  sent
    
    def tokenText(self, pos):
        """
        Adds a tokens surface form as a features.
        If pos==0: Current Token.
        If pos==-1: Previous Token
        If pos==1: Next token
        ...
        """
        index = self.sent.tokens.index(self.token)
        if 0 <= index+pos < len(self.sent.tokens):
            token = self.sent.tokens[index+pos].text
            return "token({})={}".format(pos, token)
        return "token({})=None".format(pos)
    
   

    
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
        return self.sent.tokens[0] == self.token  # Removed .text here, comparing the objects instead of the texts is more precises. Eg. if a token appears multiple times in a sentence.
    
    
    def isLast(self):
        """
        Checks if it's the last token in the sentence, excludes punct.
        """
        punct = [".", "!", "?", "..."]
        return self.token.text not in punct and self.sent.tokens[-1] == self.token  # Removed .text here, comparing the objects instead of the texts is more precises. Eg. if a token appears multiple times in a sentence.
    
   
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
        punct = [".", ",", "!", "?", "'", ";", "...", ":", "/", "``", ")", "(", "[", "]", "{", "}"]
        return self.token.text in punct
    
    def isCapitalized(self):
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
    
   
    
# General, Source Specific, Target Specific 
    def copy_features(self, token, name):
        """
        Copies all features of the token and appends the names
        """
        new_features = []
        for feature in token.features:
            new_feature = feature + "_" + name
            new_features.append(new_feature)
        token.features += new_features

    
