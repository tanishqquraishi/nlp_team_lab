
def load_onto_notes(path):
    """
    Reads a filepath and returns a list of sentences. Each sentence
    is a list of tuples. Each tuple contains a word and its POS tag.
    """

    sentences = []
    with open(path) as file:
        sentence = []
        for line in file:
            vals = line.strip().split("\t")
            if len(vals)==1:
                sentences.append(sentence)
                sentence = []
            else:
                assert(len(vals)==2)
                token, pos = vals
                sentence.append(tuple(vals))
    if len(sentence)>0:
        sentences.append(sentence)
    return sentences
