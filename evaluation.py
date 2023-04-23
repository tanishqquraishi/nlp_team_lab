from data import load_onto_notes



def confusions(gold, pred):
    conf = dict()
    for g,p in zip(gold, pred):
        if g not in conf:
            conf[g] = {"TP":0, "FP":0, "FN":0,}
        if p not in conf:
            conf[p] = {"TP":0, "FP":0, "FN":0,}
        if g==p:
            conf[g]["TP"] += 1
        else:
            conf[g]["FN"] += 1
            conf[p]["FP"] += 1
    return conf

def f1_score_micro(confusions):
    TP, FP, FN = 0,0,0
    for k,v in confusions.items():
        TP += v["TP"]
        FP += v["FP"]
        FN += v["FN"]
    P = TP / (TP+FP) if (TP+FP) > 0 else float("NaN")
    R = TP / (TP+FN) if (TP+FN) > 0 else float("NaN")
    F1 = 2*P*R / (P+R)
    return {"P":P, "R":R, "F1":F1}

def f1_score_macro(confusions):
    p_macro = 0
    r_macro = 0
    f1_macro = 0
    for v in f1_score(confusions).values():
        p_macro+=v["P"]
        r_macro+=v["R"]
        f1_macro+=v["F1"]
    p_macro /= len(confusions)
    r_macro /= len(confusions)
    f1_macro /= len(confusions)
    return {"P":p_macro,"R":r_macro,"F1":f1_macro,}


def f1_score(confusions):
    f1_scores = {}
    for k in confusions.keys():
        d = confusions[k]["TP"]+confusions[k]["FP"]
        p = confusions[k]["TP"]/d if d>0 else float("NaN")
        #
        d = confusions[k]["TP"]+confusions[k]["FN"]
        r = confusions[k]["TP"]/d if d>0 else float("NaN")
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
























