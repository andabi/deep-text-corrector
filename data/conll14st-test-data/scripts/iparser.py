# iparser.py
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

class stanfordparser:

    def __init__(self):
        pass
		
    def parse_batch(self, sentenceDumpedFileName, parsingDumpedFileName):
        
        if os.path.exists('../stanford-parser-2012-03-09') == False:
            print >> sys.stderr, 'can not find Stanford parser directory'
            sys.exit(1)
        
        # tokenized
        cmd = r'java -server -mx4096m -cp "../stanford-parser-2012-03-09/*:" edu.stanford.nlp.parser.lexparser.LexicalizedParser  -retainTMPSubcategories -sentences newline -tokenized -escaper edu.stanford.nlp.process.PTBEscapingProcessor  -outputFormat "wordsAndTags, penn, typedDependencies" -outputFormatOptions "basicDependencies" edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz ' + sentenceDumpedFileName

        r = os.popen(cmd).read().strip().decode('utf-8')
        f = open(parsingDumpedFileName, 'w')
        f.write(r.encode('utf-8'))
        f.close()

        rlist = r.replace('\n\n\n', '\n\n\n\n').split('\n\n')
        return rlist
