#!/usr/bin/env python

# This file is part of the NUS M2 scorer.
# The NUS M2 scorer is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# The NUS M2 scorer is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# file: convert_hoo.py
#
# convert source xml file and gold annotation to 
# merged file with sentence-per-line sentences and 
# annotation.
#
# usage : %prog [-p] source.xml [gold.xml] > output

from Tokenizer import PTBTokenizer
import xml.dom.minidom
import sys
import re
import getopt
from util import fix_cp1252codes


## global variables
tokenizer = PTBTokenizer()

def slice_paragraph(text):
    yield (0,len(text),text)
def slice_tokenize(text):
    import nltk
    sentence_spliter = nltk.data.load('tokenizers/punkt/english.pickle')
    last_break = 0 
    for match in sentence_spliter._lang_vars.period_context_re().finditer(text): 
        context = match.group() + match.group('after_tok') 
        if sentence_spliter.text_contains_sentbreak(context): 
            yield (last_break, match.end(), text[last_break:match.end()]) 
            if match.group('next_tok'): 
                # next sentence starts after whitespace 
                last_break = match.start('next_tok') 
            else: 
                # next sentence starts at following punctuation 
                last_break = match.end() 
    yield (last_break, len(text), text[last_break:len(text)]) 

def get_text(node):
    # get text data from xml tag
    buffer = ''
    for t in node.childNodes:
        if t.nodeType == t.TEXT_NODE:
            buffer += t.data
    return buffer

def has_empty(node):
    # check if node has <empty> tag child
    return len(node.getElementsByTagName('empty')) > 0

def get_textbody(sdom):
    parts = []
    for b in sdom.getElementsByTagName('BODY'):
        for pa in b.getElementsByTagName('PART'):
            part_id = pa.attributes["id"].value
            buffer = []
            for p in pa.getElementsByTagName('P'):
                buffer.append(get_text(p))
            parts.append((buffer, part_id))
    return parts

def get_edits(gdom):
    edits = []
    for es in gdom.getElementsByTagName('edits'):
        for e in es.getElementsByTagName('edit'):
            start = int(e.attributes["start"].value)
            end = int(e.attributes["end"].value)
            part = e.attributes["part"].value
            etype = e.attributes["type"].value
            o = e.getElementsByTagName('original')[0]
            if len(o.getElementsByTagName('empty')) > 0:
                original = ''
            else:
                original = get_text(o).strip()
            corrections = []
            optional = False
            for cs in e.getElementsByTagName('corrections'):
                for c in cs.getElementsByTagName('correction'):
                    if len(c.getElementsByTagName('empty')) > 0:
                        corrections.append('')                        
                    else:
                        correction = get_text(c).strip()
                        if correction == '':
                            optional = True
                        else:
                            corrections.append(correction)
            edits.append([start, end, part, etype, original, corrections, optional])
    return edits


# starting point
if __name__ == "__main__":
    opts, args = getopt.getopt(sys.argv[1:], "p")
    paragraph = False
    for o,a in opts :
    	if o == "-p" :
	    paragraph = True
        
    if len(args) < 1 or len(args) > 2:
        print >> sys.stderr, "usage: %prog [-p] source.xml [gold.xml] > output"
        sys.exit(-1)    
    fsource = args[0]
    gold = 0
    if len(args) == 2:
        fgold = args[1]
        gold = 1
    
    
    # parse xml files
    source_dom = xml.dom.minidom.parse(fsource)
    if gold :
        gold_dom = xml.dom.minidom.parse(fgold)    
    

    # read the xml
    parts = get_textbody(source_dom)
    if gold : 
        edits = get_edits(gold_dom)

    # sentence split
    slice = slice_tokenize
    if paragraph : 
        slice = slice_paragraph
    for part, part_no in parts:
        offset = 0
        for p in part:
            for s_start, s_end, s in slice(p):
                if s.strip() == '':
                    continue
                print "S", s.encode('utf8')
                if gold :
                   this_edits = [e for e in edits if e[0] >= offset + s_start
                                and e[1] < offset + s_end and e[2] == part_no]
                   for e in this_edits:
                      start = e[0] - (offset + s_start)
                      end = e[1] - (offset + s_start)
                      etype = e[3]
                      cor = "||".join(e[5])
                      req = "REQUIRED" if e[6] == False else "OPTIONAL"
                      out =  "A %d %d|||%s|||%s|||%s|||-NONE-|||0" % (start, end, etype, cor, req)
                      print out.encode('utf8')
                print ""
            offset += s_end 
