#-*-coding:utf-8-*-

import re
import codecs
# import nltk
# nltk.download()
# from nltk.tokenize import sent_tokenize
from hyperparams import Hyperparams as hp
from collections import namedtuple

def make_annotation_list():
    # Put tuples of features in a list.
    anns = [] # annotations
    for sent in codecs.open(hp.annotations, 'r', 'utf-8').read().split("\n\n"):
        for ann in sent.split("</MISTAKE>"):
            if "<MISTAKE" in ann:
                nid = re.search('nid="(\d+)"', ann).group(1)
                pid = re.search('pid="(\d+)"', ann).group(1)
                sid = re.search('sid="(\d+)"', ann).group(1)
                start_token = re.search('start_token="(\d+)"', ann).group(1)
                end_token = re.search('end_token="(\d+)"', ann).group(1)
                type = re.search('<TYPE>([^>]+)</TYPE>', ann).group(1)
                correction = re.search('<CORRECTION>([^>]*)</CORRECTION>', ann).group(1)
                anns.append((int(nid), int(pid), int(sid), int(start_token), int(end_token), type, correction))
    # Sort
    anns.sort(key=lambda x: (x[0], x[1], x[2], x[3], x[4]))
    
    # Filter out
    filtered = []
    ann1 = anns.pop(0)
    filtered.append(ann1)
    nid1, pid1, sid1, start_token1, end_token1, type1, correction1 = ann1
    while len(anns) > 0:
        ann2 = anns.pop(0)
        nid2, pid2, sid2, start_token2, end_token2, type2, correction2 = ann2
        
        if nid1==nid2 and pid1==pid2 and sid1==sid2: # Within the same sentence
            if start_token2 < end_token1: # Overlap
                if start_token2 >= start_token1 and end_token2 <= end_token1: # ann2 is a subset of ann1
                    continue
                elif start_token2 == start_token1 and end_token2 >= end_token1: # ann1 is a subset of ann2 
                    filtered.pop()
                else: # logical error
                    print nid2, pid2, sid2, start_token2, end_token2, type2, correction2
                    continue
                
        filtered.append(ann2) 
        nid1, pid1, sid1, start_token1, end_token1, type1, correction1 = nid2, pid2, sid2, start_token2, end_token2, type2, correction2
    
    return filtered
     
def build_corpus():
    anns = make_annotation_list()
    ann = anns.pop(0)
    print ann
    nid0, pid0, sid0, start_token0, end_token0, type0, correction0 = ann
    
    with codecs.open("../data/corpus.txt", 'w', 'utf-8') as fout:
        i=0
        with codecs.open(hp.writings, 'r', 'utf-8') as fin:
            original_sent, new_sent = [], []
            while 1:
                line = fin.readline()
                if not line: break
                
                if len(line)==1: # empty line
                    fout.write(" ".join(original_sent) + "\t" + " ".join(new_sent) + "\n")
                    original_sent, new_sent = [], []
                else:
                    nid, pid, sid, tokenid, token, _, _, _, _ = line.strip().split("\t")
                    nid, pid, sid, tokenid = int(nid), int(pid), int(sid), int(tokenid)
#                     original_sent.append(token)
                    if nid==nid0 and pid==pid0 and sid==sid0:
                        
                        if tokenid == start_token0:
#                             print line
                            original_sent.append(token)
                            new_sent.append(correction0)
                        elif start_token0 < tokenid < end_token0:
                            original_sent.append(token)
                            pass
                        else:
                            original_sent.append(token)
                            new_sent.append(token)
                        
                        if tokenid == end_token0-1:
                            if len(anns) > 0:
                                ann = anns.pop(0)
                                nid0, pid0, sid0, start_token0, end_token0, type0, correction0 = ann
                    else:
                        original_sent.append(token)
                        new_sent.append(token)
        print(i)

if __name__ == "__main__":
    build_corpus(); print "Done"