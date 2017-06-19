# nucle_doc.py
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

import os
import sys
from nltk import word_tokenize

class nucle_doc:
    def __init__(self):
        self.docattrs = None

        self.matric = ''
        self.email = ''
        self.nationality = ''
        self.firstLanguage = ''
        self.schoolLanguage = ''
        self.englishTests = ''

        self.paragraphs = []
        self.annotation = []
        self.mistakes = []

        self.sentences = []

    def buildSentence(self, sentstr, dpnode, constituentstr, poslist, chunklist):
        self.sentences[-1].append(nucle_sent(sentstr, dpnode, constituentstr, poslist, chunklist))

    def addSentence(self, sent):
        self.sentences[-1].append(sent)

    def findMistake(self, par, pos):
        for m in self.mistakes:
            if par == m['start_par'] and pos >= m['start_off'] and pos < m['end_off']:
                return m
        return None


class nucle_sent:
    def __init__(self, sentstr, dpnode, constituentstr, poslist, chunklist):
        self.sentstr = sentstr
        self.words = word_tokenize(sentstr)
        self.dpnodes = dpnode
        self.constituentstr = constituentstr
        self.constituentlist = []
        self.poslist = poslist
        self.chunklist = chunklist

    def buildConstituentList(self):

        s = self.constituentstr.strip().replace('\n', '').replace(' ', '')
        r = []
        i = 0
        while i < len(s):
            j = i
            while j < len(s) and s[j] != ')':
                j += 1
            k = j
            while k < len(s) and s[k] == ')':
                k += 1
            
            nodeWholeStr = s[i:k]
            lastLRBIndex = nodeWholeStr.rfind('(')
            nodeStr = nodeWholeStr[:lastLRBIndex] + '*' + s[j+1:k]

            r.append(nodeStr)
            i = k

        if len(r) != len(self.words):
            print >> sys.stderr, 'Error in buiding constituent tree bits: different length with words.'
            print >> sys.stderr, len(r), len(self.words)
            print >> sys.stderr, ' '.join(r).encode('utf-8')
            print >> sys.stderr, words
            sys.exit(1)

        self.constituentlist = r


    
    def setDpNode(self, dpnode):
        self.dpnodes = dpnode

    def setPOSList(self, poslist):
        self.poslist = poslist

    def setConstituentStr(self, constituentstr):
        self.constituentstr = constituentstr

    def setConstituentList(self, constituentlist):
        self.constituentlist = constituentlist

    def setWords(self, words):
        self.words = words

    def setChunkList(self, chunklist):
        self.chunklist = chunklist

    def getDpNode(self):
        return self.dpnodes

    def getPOSList(self):
        return self.poslist

    def getConstituentStr(self):
        return self.constituentstr

    def getConstituentList(self):
        return self.constituentlist 

    def getWords(self):
        return self.words
    
    def getChunkList(self):
        return self.chunklist

    def getConllFormat(self, doc, paragraphIndex, sentIndex):

        table = []

        dpnodes = self.getDpNode()
        poslist = self.getPOSList()
        #chunklist = self.getChunkList()
        words = self.getWords()
        constituentlist = self.getConstituentList()

        if len(poslist) == 0:
            hasParseInfo = 0
        else:
            hasParseInfo = 1

        if len(words) != len(poslist) and len(poslist) != 0:
            print >> sys.stderr, 'Error in buiding Conll Format: different length stanford parser postags and words.'
            print >> sys.stderr, 'len words:', len(words), words
            print >> sys.stderr, 'len poslist:', len(poslist), poslist
            sys.exit(1)

        for wdindex in xrange(len(words)):

            word = words[wdindex]

            row = []
            row.append(doc.docattrs[0][1])        #docinfo
            row.append(paragraphIndex)          #paragraph index
            row.append(sentIndex)           #paragraph index
            row.append(wdindex)             #word index
            row.append(word)                #word

            #row.append(chunknode.label)     #chunk
            if hasParseInfo == 1:

                posword = poslist[wdindex]
                splitp = posword.rfind('/')
                pos = posword[splitp+1 : ].strip()

                #chunknode = chunklist[wdindex]

                constituentnode = constituentlist[wdindex]
                
                dpnode = None
                for d in dpnodes:
                    if d.index == wdindex:
                        dpnode = d
                        break

                row.append(pos)                 #POS
                if dpnode == None:
                    row.append('-')
                    row.append('-')
                else:
                    row.append(dpnode.parent_index) #dp parent
                    row.append(dpnode.grammarrole)  #dp label
                row.append(constituentnode)         #constituent 

            table.append(row)

        return table




