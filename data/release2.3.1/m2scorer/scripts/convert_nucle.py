# convert_nucle.py
#
# Author:   Christian Hadiwinoto
#           National University of Singapore (NUS)
# Date:     12 Mar 2013
# Contact:  chrhad@comp.nus.edu.sg
#
# Version:  1.0
# 
# Original:	Yuanbin Wu
#           National University of Singapore (NUS)
# Contact:  wuyb@comp.nus.edu.sg
#
# This script is distributed to support the CoNLL-2013 Shared Task.
# It is free for research and educational purposes.
#
# Usage:   python convert_nucle.py sgmlFile > m2File

from nuclesgmlparser import nuclesgmlparser
from nucle_doc import *
import nltk.data
import re
import sys
import getopt

class PreProcessor:

    def __init__(self):

        self.sentenceTokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        self.sentenceDumpedFile = 'sentence_file'
        self.docsDumpedFileName = 'docs'

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


    def m2FileGeneration(self, docs):
        
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
                                print >> sys.stderr, "Warning: can not find word"
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
                                if wordsoffset - len(word) < m['end_off']-sentoffset: 
                                    endInWord = 1
                                    tailText = sent.sentstr[m['end_off']-sentoffset : wordsoffset].strip()
                                break

                            wordsoffset += len(word) 
                            wdend += 1
                        
                        # build annotation string for .conll.m2 file
                        m2AnnotationStr  = 'A '
                        m2AnnotationStr +=  str(wdstart) + ' '
                        m2AnnotationStr +=  str(wdend) + '|||'
                        m2AnnotationStr +=  m['type'] + '|||'
                        m2AnnotationStr +=  m['correction'].replace('\n', '') + '|||'
                        m2AnnotationStr +=  'REQUIRED|||-NONE-|||0\n'

                        m2AnnotationList.append(m2AnnotationStr)

                    # write .conll.m2 file
                    if len(m2AnnotationList) != 0:
                        m2AnnotationSent = 'S ' + tokenizedSentStr + '\n'
                        m2AnnotationSent += ''.join(m2AnnotationList) + '\n'
                        sys.stdout.write(m2AnnotationSent.encode('utf-8'))
                    

def usage_release():
    print '\nUsage: python preprocess_nmt.py sgmlFile > outputFile \n\n'

if __name__ == '__main__':
    opts, args = getopt.getopt(sys.argv[1:], "")
    
    if len(args) != 1:
        usage_release()
        sys.exit(2)

    ppr = PreProcessor()
    debug = False
   
    sgmlFileName = args[0]

    docs = ppr.sentenceSplit(ppr.readNUCLE(sgmlFileName))
    ppr.m2FileGeneration(docs)
