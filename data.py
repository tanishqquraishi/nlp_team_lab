
def load_onto_notes(path):
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
