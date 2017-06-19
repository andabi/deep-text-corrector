#!/usr/bin/python

# preprocesswithalt.py
#
# Author:	Christian Hadiwinoto
#           National University of Singapore (NUS)
# Date:		22 Apr 2014
# Version:      1.0
# 
# Contact:  chrhad@comp.nus.edu.sg
#
# This script is distributed to support the CoNLL-2013 Shared Task.
# It is free for research and educational purposes.
#
# Usage:   python preprocesswithalt.py essaySgmlFileName M mainSgmlFileName alt1SgmlFileName ... altNSgmlFileName m2FileName
#


import parser_feature
from nuclesgmlparser import nuclesgmlparser
from nucle_doc import *
import nltk.data
from nltk import word_tokenize
from operator import itemgetter
import cPickle as pickle
import re
import sys
import os

getEditKey = itemgetter(0, 1, 2, 3, 4)

sentenceTokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
sentenceDumpedFile = 'sentence_file'
docsDumpedFileName = 'docs'
parsingDumpedFileName = 'parse_file'

def readNUCLE(fn):

    f = open(fn, 'r')
    parser = nuclesgmlparser()
    filestr = f.read()
    filestr = filestr.decode('utf-8')
    
    #Fix Reference tag
    p = re.compile(r'(<REFERENCE>\n<P>\n.*\n)<P>')
    filestr = p.sub(r'\1</P>', filestr)

    parser.feed(filestr)
    f.close()
    parser.close()

    return parser.docs

def sentenceSplit(docs):

    sentenceTokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    for doc in docs:
        for par in doc.paragraphs:
            doc.sentences.append([])
            for s in sentenceTokenizer.tokenize(par):
                doc.buildSentence(s, [], '', [], [])
    return docs
 
def compareTwoEditLists(editList1, editList2):
    # must be sorted
    if editList1 == [] and editList2 == []:
        return True
    elif editList1 == [] or editList2 == []:
        return False
    elif getEditKey(editList1[0]) != getEditKey(editList2[0]):
        return False
    else:
        return compareTwoEditLists(editList1[1:], editList2[1:])

def moderateAnnotations(contestDocs, annotBoard, origDocSet):
    # moderate annotation in "contesting" docs with already stated mistakes
    #mistakeStrSet = {}
    for doc in contestDocs:
        #mistakeStr = ''
        nid = int(doc.docattrs[0][1]) # nid of current document
        tid = doc.annotation[0][0][1] # teacher id

        if not annotBoard.has_key(nid): # create placeholder
            annotBoard[nid] = {}

        origDoc = origDocSet[nid]
        for pid in xrange(len(origDoc.sentences)):
            slist = origDoc.sentences[pid]
            if not annotBoard[nid].has_key(pid):
                annotBoard[nid][pid] = {}
            for sentid in xrange(len(slist)):
                sent = slist[sentid]
                if not annotBoard[nid][pid].has_key(sentid):
                    annotBoard[nid][pid][sentid] = []
                editSet = []

                # enumerate mistakes
                sentoffset = origDoc.paragraphs[pid].index(sent.sentstr)
                editNum = 0
                for m in doc.mistakes:
                    if m['start_par'] != pid or \
                       m['start_par'] != m['end_par'] or \
                       m['start_off'] < sentoffset or \
                       m['start_off'] >= sentoffset + len(sent.sentstr) or \
                       m['end_off'] <sentoffset or \
                       m['end_off'] > sentoffset + len(sent.sentstr):
                        continue

                    #if m['type'] != 'noop':
                    editSet.append((m['start_par'], m['end_par'], m['start_off'], m['end_off'], m['correction'], m['type']))
                        #editNum += 1
                    #else:
                        #editSet.append((m['start_par'], m['end_par'], m['start_off'], m['end_off'], sent.sentstr, m['type']))

                editSet = sorted(editSet, key=itemgetter(0, 1, 2, 3))
                
                # find the same annotation
                foundMatch = False
                i = 0
                boardEdits = annotBoard[nid][pid][sentid]
                while i < len(boardEdits) and not foundMatch:
                    if compareTwoEditLists(editSet, boardEdits[i]):
                        foundMatch = True
                    else:
                        i+=1

                if not foundMatch:
                    annotBoard[nid][pid][sentid].append(editSet)
        
    return annotBoard

def moderateAnnotationsAlt(contestDocs, annotBoard, origDocSet):
    # moderate annotation in "contesting" docs with already stated mistakes
    # for alternative answers (with explicit NOOP)
    mistakeStrSet = {}
    for doc in contestDocs:
        mistakeStr = ''
        nid = int(doc.docattrs[0][1]) # nid of current document
        tid = doc.annotation[0][0][1] # teacher id

        if not annotBoard.has_key(nid): # create placeholder
            annotBoard[nid] = {}

        origDoc = origDocSet[nid]
        for pid in xrange(len(origDoc.sentences)):
            slist = origDoc.sentences[pid]
            if not annotBoard[nid].has_key(pid):
                annotBoard[nid][pid] = {}
            for sentid in xrange(len(slist)):
                sent = slist[sentid]
                if not annotBoard[nid][pid].has_key(sentid):
                    annotBoard[nid][pid][sentid] = []
                editSet = []

                # enumerate mistakes
                sentoffset = origDoc.paragraphs[pid].index(sent.sentstr)
                editNum = 0
                for m in doc.mistakes:
                    if m['start_par'] != pid or \
                       m['start_par'] != m['end_par'] or \
                       m['start_off'] < sentoffset or \
                       m['start_off'] >= sentoffset + len(sent.sentstr) or \
                       m['end_off'] <sentoffset or \
                       m['end_off'] > sentoffset + len(sent.sentstr):
                        continue

                    if m['type'] != 'noop':
                        editSet.append((m['start_par'], m['end_par'], m['start_off'], m['end_off'], m['correction'], m['type']))
                        editNum += 1
                    else:
                        editSet.append((m['start_par'], m['end_par'], m['start_off'], m['end_off'], sent.sentstr, m['type']))
                
                # as empty alternative edit means agreement to main annotation edit
                if len(editSet) == 0:
                    continue

                editSet = sorted(editSet, key=itemgetter(0, 1, 2, 3))
                
                # find the same annotation
                foundMatch = False
                i = 0
                boardEdits = annotBoard[nid][pid][sentid]
                while i < len(boardEdits) and not foundMatch:
                    if compareTwoEditLists(editSet, boardEdits[i]):
                        foundMatch = True
                    else:
                        i+=1

                if not foundMatch:
                    annotBoard[nid][pid][sentid].append(editSet)
        
    return annotBoard
    
def createM2File(origDocs, mistakesBoard, m2FileName):
    
    fm2 = open(m2FileName, 'w')

    for doc in origDocs:
        nid = int(doc.docattrs[0][1]) # nid of current document
        for slistIndex in xrange(len(doc.sentences)):
            slist = doc.sentences[slistIndex]
            for sentid in xrange(len(slist)):

                sent = slist[sentid]

                # m2 format annotation string list
                m2AnnotationList = []

                # build colums
                table = sent.getConllFormat(doc, slistIndex, sentid)
                tokenizedSentStr = ' '.join(sent.getWords())

                #Add annotation info
                sentoffset = doc.paragraphs[slistIndex].index(sent.sentstr)

                i = 0
                board = mistakesBoard[nid][slistIndex][sentid]
                for mistakesList in board:
                    if len(mistakesList) == 0 and len(board) > 1:
                        m2AnnotationList.append('A -1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||' + str(i) + '\n')
                        i += 1
                        
                    for tuple in mistakesList:
                        m = {}
                        m['start_par'] = tuple[0]
                        m['end_par'] = tuple[1]
                        m['start_off'] = tuple[2]
                        m['end_off'] = tuple[3]
                        m['correction'] = tuple[4]
                        m['type'] = tuple[5]

                        if m['start_par'] != slistIndex or \
                          m['start_par'] != m['end_par'] or \
                          m['start_off'] < sentoffset or \
                          m['start_off'] >= sentoffset + len(sent.sentstr) or \
                          m['end_off'] <sentoffset or \
                          m['end_off'] > sentoffset + len(sent.sentstr):
                            continue

                        wordsoffset = 0
                        wdstart = 0

                        startInWord = 0
                        headText = ''
                        endInWord = 0
                        tailText = ''

                        words = sent.getWords()
                        while wdstart < len(words):

                            word = words[wdstart]
                            nextstart = sent.sentstr.find(word, wordsoffset)

                            if nextstart == -1:
                                # may not find word, due to relpacement
                                print >> sys.stderr, "Warning in building conll format: can not find word"
                                print >> sys.stderr, word.encode('utf-8')
                                wordsoffset += 1
                            else:
                                wordsoffset = nextstart

                            if wordsoffset >= m['start_off']-sentoffset:
                                break
                            elif wordsoffset + len(word) > m['start_off']-sentoffset:
                                # annotation starts at the middle of a word
                                startInWord = 1
                                headText = sent.sentstr[wordsoffset: m['start_off']-sentoffset]
                                break

                            wordsoffset += len(word) 
                            wdstart += 1

                        if wdstart == len(words):
                            print >> sys.stderr, 'Warning in building conll format: start_off overflow'
                            print >> sys.stderr, m, sent.sentstr.encode('utf-8')
                            continue


                        wdend = wdstart
                        while wdend < len(words):

                            word = words[wdend]
                            
                            nextstart = sent.sentstr.find(word, wordsoffset)

                            if nextstart == -1:
                                print >> sys.stderr, "Warning in building conll format: can not find word"
                                print >> sys.stderr, word.encode('utf-8')
                                wordsoffset += 1
                            else:
                                wordsoffset = nextstart

                            if wordsoffset >= m['end_off']-sentoffset:
                                # annotation ends at the middle of a word
                                if wordsoffset - len(words[wdend-1]) - 1 < m['end_off']-sentoffset: 
                                    endInWord = 1
                                    tailText = sent.sentstr[m['end_off']-sentoffset : wordsoffset].strip()
                                break

                            wordsoffset += len(word) 
                            wdend += 1
                       

                        correctionTokenizedStr = tokenizeCorrectionStr(headText + m['correction'] + tailText, wdstart, wdend, words)
                        correctionTokenizedStr, wdstart, wdend = shrinkCorrectionStr(correctionTokenizedStr, wdstart, wdend, words)

                        token_start = wdstart if m['type'] != 'noop' else -1
                        token_end = wdend if m['type'] != 'noop' else -1
                        correction_final = correctionTokenizedStr.replace('\n', '') if m['type'] != 'noop' else '-NONE-'

                        # build annotation string for .conll.m2 file
                        m2AnnotationStr  = 'A '
                        m2AnnotationStr +=  str(token_start) + ' '
                        m2AnnotationStr +=  str(token_end) + '|||'
                        m2AnnotationStr +=  m['type'] + '|||'
                        m2AnnotationStr +=  correction_final + '|||'
                        m2AnnotationStr +=  'REQUIRED|||-NONE-|||' + str(i) + '\n'

                        m2AnnotationList.append(m2AnnotationStr)
                    
                    if len(mistakesList) > 0: # only if mistakeList contains tuples
                        i += 1

                # write .conll.m2 file
                m2AnnotationSent = 'S ' + tokenizedSentStr + '\n'
                m2AnnotationSent += ''.join(m2AnnotationList) + '\n'
                fm2.write(m2AnnotationSent.encode('utf-8'))
    
    fm2.close()


def tokenizeCorrectionStr(correctionStr, wdstart, wdend, words):

    correctionTokenizedStr = ''
    pseudoSent = correctionStr

    if wdstart != 0:
        pseudoSent = words[wdstart-1] + ' ' + pseudoSent 

    if wdend < len(words) - 1:
        pseudoSent = pseudoSent + ' ' + words[wdend]
    elif wdend == len(words) - 1:
        pseudoSent = pseudoSent + words[wdend]


    pseudoWordsList = []
    sentList = sentenceTokenizer.tokenize(pseudoSent)
    for sent in sentList:
        pseudoWordsList += word_tokenize(sent)

    start = 0
    if wdstart != 0:
        s = ''
        for i in xrange(len(pseudoWordsList)):
            s += pseudoWordsList[i]
            if s == words[wdstart-1]:
                start = i + 1
                break
        if start == 0:
            print >> sys.stderr, 'Can not find words[wdstart-1]'

    else:
        start = 0

    end = len(pseudoWordsList)
    if wdend != len(words):

        s = ''
        for i in xrange(len(pseudoWordsList)):
            s = pseudoWordsList[len(pseudoWordsList) - i - 1] + s
            if s == words[wdend]:
                end = len(pseudoWordsList) - i - 1
                break
        if end == len(pseudoWordsList):
            print >> sys.stderr, 'Can not find words[wdend]'

    else:
        end = len(pseudoWordsList)

    correctionTokenizedStr = ' '.join(pseudoWordsList[start:end])

    return correctionTokenizedStr


def shrinkCorrectionStr(correctionTokenizedStr, wdstart, wdend, words):

    correctionWords = correctionTokenizedStr.split(' ')
    originalWords = words[wdstart: wdend]
    wdstartNew = wdstart
    wdendNew = wdend
    cstart = 0
    cend = len(correctionWords)

    i = 0
    while i < len(originalWords) and i < len(correctionWords):
        if correctionWords[i] == originalWords[i]:
            i += 1
            wdstartNew = i + wdstart
            cstart = i
        else:
            break

    i = 1
    while i <= len(originalWords) - cstart and i <= len(correctionWords) - cstart:
        if correctionWords[len(correctionWords)-i] == originalWords[len(originalWords)-i]:
            wdendNew = wdend - i
            cend = len(correctionWords) - i
            i += 1
        else:
            break

    return ' '.join(correctionWords[cstart:cend]), wdstartNew, wdendNew

if __name__ == '__main__':

    ''' usage: 

        %python preprocesswithalt.py essaySgmlfile M mainsgmlfile1 ... mainsgmlfileM alternativesgmlfile1 ... alternativesgmlfileN combinedm2file
          output an m2 file containing a collection of M main annotations and N alternative annotations.

        In most cases essaySgmlfile and mainsgmlfile are identical
    '''

    # Load original complete SGML for reference
    origDocs = sentenceSplit(readNUCLE(sys.argv[1]))
    
    origDocSet = {}
    for doc in origDocs:
        nid = int(doc.docattrs[0][1])
        origDocSet[nid] = doc

    nummain = int(sys.argv[2])
    
    # Store main annotations
    docsList = []
    altDocsList = []
    for i in range(0, nummain):
        print >> sys.stderr, 'Storing main annotation', (i+1)
        docs = sentenceSplit(readNUCLE(sys.argv[i+3]))
        docsList.append(docs)

    board = {}
    for docs in docsList:
        board = moderateAnnotations(docs, board, origDocSet)
    
    # store alternative annotations
    for i in range(3 + nummain, len(sys.argv) - 1):
        print >> sys.stderr, 'Storing alternative annotation', (i+1)
        altdocs = sentenceSplit(readNUCLE(sys.argv[i]))
        altDocsList.append(altdocs)

    for altdocs in altDocsList:
        board = moderateAnnotationsAlt(altdocs, board, origDocSet)

    createM2File(origDocs, board, sys.argv[len(sys.argv)-1])

    pass

