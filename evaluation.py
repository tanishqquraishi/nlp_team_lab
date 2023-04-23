from data import load_onto_notes

def confusions(gold, pred: list) -> dict:
    """
    Creates a confusion matrix with a list of gold and predicted labels.
    """
    conf = dict()       # Empty dict to store true positives (TP), false positives (FP), false negatives (FN) for each label
    for g,p in zip(gold, pred):
        if g not in conf:   
            conf[g] = {"TP":0, "FP":0, "FN":0,}
        if p not in conf:
            conf[p] = {"TP":0, "FP":0, "FN":0,}
        if g==p:        # If gold and pred match, increment TP count
            conf[g]["TP"] += 1
        else:           # If gold and pred mismatch, increment FN and FP counts
            conf[g]["FN"] += 1
            conf[p]["FP"] += 1
    return conf

def f1_score_micro(confusions: dict) -> dict:
    """
    Computes the micro-averaged F1 score using counts of true positives (TP), false positives (FP)
    and false negatives (FN) for each label.
    """
    
    TP, FP, FN = 0,0,0
    for k,v in confusions.items():
        TP += v["TP"]       # Increment TP with value of TP in dict
        FP += v["FP"]       # Increment FP with value of TP in dict
        FN += v["FN"]       # Increment FN with value of TP in dict
    P = TP / (TP+FP) if (TP+FP) > 0 else float("NaN")   # Computes precision
    R = TP / (TP+FN) if (TP+FN) > 0 else float("NaN")   # Computes recall 
    F1 = 2*P*R / (P+R)
    return {"P":P, "R":R, "F1":F1}

def f1_score_macro(confusions: dict) -> dict:
    """
    Computes the macro F1 score using precision, recall and F1 scores.
    """
    
    p_macro = 0     # Initialize macro precision
    r_macro = 0     # Initialize macro recall
    f1_macro = 0    # Initialize macro F1
    for v in f1_score(confusions).values():
        p_macro+=v["P"]
        r_macro+=v["R"]
        f1_macro+=v["F1"]
    p_macro /= len(confusions)  
    r_macro /= len(confusions)
    f1_macro /= len(confusions)
    return {"P":p_macro,"R":r_macro,"F1":f1_macro,}


def f1_score(confusions: dict) -> dict:
    """
    Computes F1 score for each class i.e. each part of speech.
    """
    
    f1_scores = {}
    for k in confusions.keys():
        d = confusions[k]["TP"]+confusions[k]["FP"]         # Computes the denominator
        p = confusions[k]["TP"]/d if d>0 else float("NaN")  # Computes precision, returns NaN if denominator is 0
        #
        d = confusions[k]["TP"]+confusions[k]["FN"]
        r = confusions[k]["TP"]/d if d>0 else float("NaN")  # Computes recall, returns NaN if denominator is 0
        ##
        d = p+r
        f1 = 2*p*r/d
        f1_scores[k] = {"P":p, "R":r, "F1":f1}
    return f1_scores


a = load_onto_notes("../pos_tagger_data/train.col")
_,a = zip(*a[0])

print("MACRO:")
print(f1_score_macro(confusions(a,a)))
print("MICRO:")
print(f1_score_micro(confusions(a,a)))
print("No Avg.:")
print(f1_score(confusions(a,a)))
























