# parser_feature.py
#
# Author:	Yuanbin Wu
#           National University of Singapore (NUS)
# Date:		12 Mar 2013
# Version:      1.0
# 
# Contact:  wuyb@comp.nus.edu.sg
#
# This script is distributed to support the CoNLL-2013 Shared Task.
# It is free for research and educational purposes.



import iparser

class stanpartreenode:
    def __init__(self, strnode):

        if strnode == '':
            self.grammarrole = ''
            self.parent_index = -1
            self.index = -1
            self.parent_word = ''
            self.word = ''
            self.POS = ''
            return

        groleend = strnode.find('(')
        self.grammarrole = strnode[ : groleend]
        content = strnode[groleend + 1: len(strnode)-1]
        dadAndme = content.partition(', ')
        dad = dadAndme[0]
        me = dadAndme[2]
        dadsep = dad.rfind('-')
        mesep = me.rfind('-')
        self.parent_index = int(dad[dadsep + 1 : ]) - 1 
        self.parent_word = dad[0 : dadsep]
        self.index = int(me[mesep + 1 : ]) - 1
        self.word = me[0 : mesep]
        self.POS = '' 

        
def DependTree_Batch(sentenceDumpedFileName, parsingDumpedFileName):

    sparser = iparser.stanfordparser()
    results = sparser.parse_batch(sentenceDumpedFileName, parsingDumpedFileName)
    nodeslist = []

    k = 0
    while k < len(results):
        PoSlist = results[k].split(' ')
        constituentstr = results[k+1]
        table = results[k+2].split('\n')
        nodes = []
        for i in range(0, len(table)):
            nodes.append( stanpartreenode(table[i]) )
        nodeslist.append((nodes, constituentstr, PoSlist))
        k += 3
    return nodeslist

def DependTree_Batch_Parsefile(parsingDumpedFileName):

    f = open(parsingDumpedFileName, 'r')
    results = f.read().decode('utf-8').replace('\n\n\n', '\n\n\n\n').split('\n\n')
    f.close()
    nodeslist = []

    k = 0
    while k < len(results):
        PoSlist = results[k].split(' ')
        constituentstr = results[k+1]
        table = results[k+2].split('\n')

        nodes = []
        for i in range(0, len(table)):
            nodes.append( stanpartreenode(table[i]) )
        nodeslist.append((nodes, constituentstr, PoSlist))
        k += 3
    return nodeslist
