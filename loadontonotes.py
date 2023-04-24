class LoadOntoNotes:
    """
    Takes a path, reads the file and returns token and POS pairs 
    """
    def __init__(self, path):
        self.path = path 

    def read_file(self):
        corpus = open(self.path)
        return corpus 
    
    def get_tokens(self):
        sentences = []
        tok_pos_pairs = []
        for lines in self.read_file():
            toks = lines.strip().split("\t")
            if len(toks) == 2:  
                tok_pos_pairs.append(toks)
            else:
                sentences.append(tok_pos_pairs)
                tok_pos_pairs = []
        if tok_pos_pairs:
            sentences.append(tok_pos_pairs)
        return sentences


lon = LoadOntoNotes("train.col")
print(lon.get_tokens())