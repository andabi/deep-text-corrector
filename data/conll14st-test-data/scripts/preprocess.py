#!/usr/bin/python

# preprocess_nmt.py
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
#
# Usage:   python preprocess_nmt.py OPTIONS sgmlFileName conllFileName annotationFileName m2FileName
# Options:
#         -o    generate conllFile, annotationFile, m2File from sgmlFile, with parser info. 
#         -l    generate conllFile, annotationFile, m2File from sgmlFile, without parser info.


import parser_feature
from nuclesgmlparser import nuclesgmlparser
from nucle_doc import *
import nltk.data
from nltk import word_tokenize
import cPickle as pickle
import re
import sys
import os
import getopt

class PreProcessor:

    def __init__(self):

        self.sentenceTokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        self.sentenceDumpedFile = 'sentence_file'
        self.docsDumpedFileName = 'docs'
        self.parsingDumpedFileName = 'parse_file'

    def readNUCLE(self, fn):

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


    def sentenceSplit(self, docs):

        for doc in docs:
            for par in doc.paragraphs:
                doc.sentences.append([])
                for s in self.sentenceTokenizer.tokenize(par):
                    doc.buildSentence(s, [], '', [], [])
        return docs
 

    def featureGeneration(self, docs, option):

        # build parsing feature
        # the sentence for parsing is dump to self.sentenceDumpedFile
        f = open(self.sentenceDumpedFile, 'w')

        for doc in docs:
            for par in doc.paragraphs:
                doc.sentences.append([])
                for s in self.sentenceTokenizer.tokenize(par):
                    sent = nucle_sent(s, [], '', [], [])
                    doc.addSentence(sent)
                    tokenizedSentStr = ' '.join(sent.getWords()) + '\n'

                    f.write(tokenizedSentStr.encode('utf-8'))
        f.close()

        if option == 0:
            nodelist = parser_feature.DependTree_Batch(self.sentenceDumpedFile, self.parsingDumpedFileName)
        elif option == 1 :
            nodelist = parser_feature.DependTree_Batch_Parsefile(self.parsingDumpedFileName)
        else:
            return

        i = 0
        for doc in docs:
            for slist in doc.sentences:
                for s in slist:
                    if s.sentstr.strip() == '':
                        continue

                    s.setDpNode(nodelist[i][0])
                    s.setConstituentStr(nodelist[i][1])
                    s.setPOSList(nodelist[i][2])
                    s.buildConstituentList()

                    i += 1

        f = file(self.docsDumpedFileName,'w')
        pickle.dump(docs, f)
        f.close()
        return docs


    def conllFileGeneration(self, docs, conllFileName, annotationFileName, m2FileName):
        
        fcolumn = open(conllFileName, 'w')
        fannotation = open(annotationFileName, 'w')
        fm2 = open(m2FileName, 'w')

        for doc in docs:
            for slistIndex in xrange(len(doc.sentences)):
                slist = doc.sentences[slistIndex]
                for sentid in xrange(len(slist)):

                    sent = slist[sentid]

                    # annotation string list
                    annotationList = []

                    # m2 format annotation string list
                    m2AnnotationList = []

                    # build colums
                    table = sent.getConllFormat(doc, slistIndex, sentid)
                    tokenizedSentStr = ' '.join(sent.getWords())

                    #Add annotation info
                    sentoffset = doc.paragraphs[slistIndex].index(sent.sentstr)
                    for m in doc.mistakes:

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
                       

                        correctionTokenizedStr = self.tokenizeCorrectionStr(headText + m['correction'] + tailText, wdstart, wdend, words)
                        
                        #Shrink the correction string, wdstart, wdend
                        correctionTokenizedStr, wdstart, wdend = self.shrinkCorrectionStr(correctionTokenizedStr, wdstart, wdend, words)
                        if wdstart == wdend and len(correctionTokenizedStr) == 0:
                            continue

                        # build annotation string for .conll.ann file
                        annotationStr  = '<MISTAKE '
                        annotationStr += 'nid="' + table[0][0] + '" '          #nid
                        annotationStr += 'pid="' + str(table[0][1]) + '" '       #start_par
                        annotationStr += 'sid="' + str(sentid) + '" '                   #sentence id
                        annotationStr += 'start_token="' + str(wdstart) + '" '          #start_token
                        annotationStr += 'end_token="' + str(wdend) + '">\n'            #end_token
                        annotationStr += '<TYPE>' + m['type'] + '</TYPE>\n'
                        annotationStr += '<CORRECTION>' + correctionTokenizedStr + '</CORRECTION>\n'
                        annotationStr += '</MISTAKE>\n'

                        annotationList.append(annotationStr)

                        # build annotation string for .conll.m2 file
                        m2AnnotationStr  = 'A '
                        m2AnnotationStr +=  str(wdstart) + ' '
                        m2AnnotationStr +=  str(wdend) + '|||'
                        m2AnnotationStr +=  m['type'] + '|||'
                        m2AnnotationStr +=  correctionTokenizedStr.replace('\n', '')  + '|||'
                        m2AnnotationStr +=  'REQUIRED|||-NONE-|||0\n'

                        m2AnnotationList.append(m2AnnotationStr)



                    # write .conll file
                    for row in table:
                        output = ''
                        for record in row:
                            if type(record) == type(1):
                                output = output + str(record) + '\t'
                            else:
                                output = output + record + '\t'
                        fcolumn.write((output.strip() + '\n').encode('utf-8'))
                    fcolumn.write(('\n').encode('utf-8'))

                    # write .conll.ann file
                    if len(annotationList) != 0:
                        annotationSent = '<ANNOTATION>\n' + ''.join(annotationList) + '</ANNOTATION>\n'
                        fannotation.write((annotationSent + '\n').encode('utf-8'))

                    # write .conll.m2 file
                    m2AnnotationSent = 'S ' + tokenizedSentStr + '\n'
                    m2AnnotationSent += ''.join(m2AnnotationList) + '\n'
                    fm2.write(m2AnnotationSent.encode('utf-8'))
                    
        fcolumn.close()
        fannotation.close()
        fm2.close()


    def tokenizeCorrectionStr(self, correctionStr, wdstart, wdend, words):

        correctionTokenizedStr = ''
        pseudoSent = correctionStr

        if wdstart != 0:
            pseudoSent = words[wdstart-1] + ' ' + pseudoSent 

        if wdend < len(words) - 1:
            pseudoSent = pseudoSent + ' ' + words[wdend]
        elif wdend == len(words) - 1:
            pseudoSent = pseudoSent + words[wdend]


        pseudoWordsList = []
        sentList = self.sentenceTokenizer.tokenize(pseudoSent)
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


    def shrinkCorrectionStr(self, correctionTokenizedStr, wdstart, wdend, words):

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


        

def usage_debug():

    u = '\nUsage: python preprocess_nmt.py options \n\n'
    u += '-g sgmlFileName -d useDumpedFile\n'
    u += '   generate sentence features and dump the results \n'
    u += '   sgmlFileName: the nucle sgml files \n'
    u += '   useDumpedFile = 0, don\'t use dumped files, will parse nucle sgml file (Default) \n'
    u += '   useDumpedFile = 1, reuse previous dumped parse files \n\n'
    u += '-c conllFileName annotationFileName m2FileName \n' 
    u += '   generate conllFile, annotationFile, m2File \n\n'
    u += '-l sgmlFileName conllFileName annotationFileName m2FileName \n' 
    u += '   generate conllFile, annotationFile, m2FileName from sgmlFile, without parser info.\n'
    print u

def usage_release():
    u = '\nUsage: python preprocess_nmt.py OPTIONS sgmlFileName conllFileName annotationFileName m2FileName \n\n'
    u += '-o    generate conllFile, annotationFile, m2File from sgmlFile, with parser info.\n' 
    u += '-l    generate conllFile, annotationFile, m2File from sgmlFile, without parser info.\n'   
    print u



if __name__ == '__main__':

    ppr = PreProcessor()
    debug = False
    try:
        if debug == True:
            opts, args = getopt.getopt(sys.argv[1:],'g:d:c:l:h')
        else:
            opts, args = getopt.getopt(sys.argv[1:],'l:o:h')
    except getopt.GetoptError:

        if debug == True:
            usage_debug()
        else: 
            usage_release()
        sys.exit(2)
   
    option = {}
    option['-g'] = 0
    option['-c'] = 0
    option['-l'] = 0
    option['-o'] = 0
    option['useDumpedFile'] = 0
    option['sgmlFileName'] = None
    option['conllFileName'] = None
    option['annotationFileName'] = None
    option['m2FileName'] = None


    for opt, arg in opts:
        if opt == '-g':
            if os.path.isfile(arg) == False:
                print >> sys.stderr, 'can not find sgml file'
                sys.exit(2)
            else:
                option['sgmlFileName'] = arg
                option['-g'] = 1

        elif opt == '-d':
            if arg not in ('1', '0'):
                print >> sys.stderr, '-d option should be 0 or 1'
                sys.exit(2)
            else:
                option['useDumpedFile'] = int(arg)

        elif opt == '-c':
            if len(args) != 2:
                print >> sys.stderr, '-c option need 3 args'
                sys.exit(2)
            else:
                option['conllFileName'] = arg
                option['annotationFileName'] = args[0]
                option['m2FileName'] = args[1]
                option['-c'] = 1

        elif opt == '-l':
            if len(args) != 3:
                print >> sys.stderr, '-l option need 4 args'
                sys.exit(2)
            else:
                if os.path.isfile(arg) == False:
                    print >> sys.stderr, 'can not find sgml file'
                    sys.exit(2)
                else:
                    option['sgmlFileName'] = arg

                option['conllFileName'] = args[0]
                option['annotationFileName'] = args[1]
                option['m2FileName'] = args[2]
                option['-l'] = 1


        elif opt == '-o':
            if len(args) != 3:
                print >> sys.stderr, '-o option need 4 args'
                sys.exit(2)
            else:
                if os.path.isfile(arg) == False:
                    print >> sys.stderr, 'can not find sgml file'
                    sys.exit(2)
                else:
                    option['sgmlFileName'] = arg

                option['conllFileName'] = args[0]
                option['annotationFileName'] = args[1]
                option['m2FileName'] = args[2]
                option['useDumpedFile'] = 0
                option['-o'] = 1

        elif opt == '-h':
            if debug == True:
                usage_debug()
            else:
                usage_release()
            sys.exit()


    if option['-g'] + option['-c'] + option['-l'] + option['-o']  > 1:
        print >> sys.stderr, 'only one option among -g, -c, -l, -o is allowed'
        sys.exit(2)
    elif option['-g'] + option['-c'] + option['-l'] + option['-o'] == 0:
        print >> sys.stderr, 'no option given'
        sys.exit(2)
    
           
    if option['-g'] == 1:
        docs = ppr.readNUCLE(option['sgmlFileName'])
        ppr.featureGeneration(docs, option['useDumpedFile'])

    elif option['-c']  == 1:
        if os.path.isfile(ppr.docsDumpedFileName) == False:
            print >> sys.stderr, '-c option needs dumped \'docs\' file. Please use -g option first. '
            sys.exit(2)
        f = file(ppr.docsDumpedFileName, 'r')
        docs = pickle.load(f)
        f.close()

        ppr.conllFileGeneration(docs, option['conllFileName'], option['annotationFileName'], option['m2FileName'])

    elif option['-l'] == 1:
        docs = ppr.sentenceSplit(ppr.readNUCLE(option['sgmlFileName']))
        ppr.conllFileGeneration(docs, option['conllFileName'], option['annotationFileName'], option['m2FileName'])

    elif option['-o'] == 1:

        docs = ppr.readNUCLE(option['sgmlFileName'])
        docs = ppr.featureGeneration(docs, 0)
        ppr.conllFileGeneration(docs, option['conllFileName'], option['annotationFileName'], option['m2FileName'])

        os.remove(ppr.sentenceDumpedFile)
        os.remove(ppr.docsDumpedFileName)
        os.remove(ppr.parsingDumpedFileName)

