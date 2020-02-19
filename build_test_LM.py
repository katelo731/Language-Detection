#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import re
import nltk
import sys
import getopt
import math
import string

n = 4                               # n-grams
k = 1                               # add-k smoothing
is_case_folding = True
remove_punctuation = True
add_padding = False
normalize_probability = True
exist_other_language = True
threshold = 1e-100
ngram_list = {}                     # record all n-grams

def build_LM(in_file):
    """
    build language models for each label
    each line in in_file contains a label and a string separated by a space
    """
    print('building language models...')
    
    # initialize
    ngram_dict = [[0, {}], [0, {}], [0, {}]]     # use dictionary to store ngrams and their counts
    ngram_prob = [[0, {}], [0, {}], [0, {}]]     # use dictionary to store ngrams and their probability
    
    # read in_file
    inf = open(in_file, encoding="utf8")
    for line in inf:
        # seperate language type and sentence for each line
        language, sentence = line.split(' ', 1)
        
        # preprocess the sentence
        if is_case_folding:
            sentence = sentence.lower()
        if remove_punctuation:
            "".join(l for l in sentence if l not in string.punctuation)
        if add_padding:
            sentence = "!!!" + sentence + "$$$"
        
        # go thru the entire sentence
        for i in range(len(sentence) - n):
            ngram = sentence[i : i + n]
            # create a new key in dictionary
            if ngram not in ngram_list:
                ngram_list[ngram] = 1
                if language == "malaysian":
                    ngram_dict[0][1][ngram] = 1 + k
                    ngram_dict[1][1][ngram] = k
                    ngram_dict[2][1][ngram] = k
                    ngram_dict[0][0] += 1 + k
                    ngram_dict[1][0] += k
                    ngram_dict[2][0] += k
                elif language == "indonesian":
                    ngram_dict[0][1][ngram] = k
                    ngram_dict[1][1][ngram] = 1 + k
                    ngram_dict[2][1][ngram] = k
                    ngram_dict[0][0] += k
                    ngram_dict[1][0] += 1 + k
                    ngram_dict[2][0] += k
                elif language == "tamil":
                    ngram_dict[0][1][ngram] = k
                    ngram_dict[1][1][ngram] = k
                    ngram_dict[2][1][ngram] = 1 + k
                    ngram_dict[0][0] += k
                    ngram_dict[1][0] += k
                    ngram_dict[2][0] += 1 + k
            # ngram already exists
            else:
                ngram_list[ngram] += 1
                if language == "malaysian":
                    ngram_dict[0][1][ngram] += 1
                    ngram_dict[0][0] += 1
                elif language == "indonesian":
                    ngram_dict[1][1][ngram] += 1
                    ngram_dict[1][0] += 1
                elif language == "tamil":
                    ngram_dict[2][1][ngram] += 1
                    ngram_dict[2][0] += 1
    
    # build prob model
    for ngram in ngram_list.keys():
        if normalize_probability:
            ngram_prob[0][1][ngram] = -1 / math.log2(float(ngram_dict[0][1][ngram] / ngram_dict[0][0]))
            ngram_prob[1][1][ngram] = -1 / math.log2(float(ngram_dict[1][1][ngram] / ngram_dict[1][0]))
            ngram_prob[2][1][ngram] = -1 / math.log2(float(ngram_dict[2][1][ngram] / ngram_dict[2][0]))
        else:
            ngram_prob[0][1][ngram] = float(ngram_dict[0][1][ngram] / ngram_dict[0][0])
            ngram_prob[1][1][ngram] = float(ngram_dict[1][1][ngram] / ngram_dict[1][0])
            ngram_prob[2][1][ngram] = float(ngram_dict[2][1][ngram] / ngram_dict[2][0])
    
    return (ngram_prob)

def test_LM(in_file, out_file, LM):
    """
    test the language models on new strings
    each line of in_file contains a string
    you should print the most probable label for each string into out_file
    """
    print("testing language models...")
    
    ngram_prob = LM
    
    inf = open(in_file, 'r')
    outf = open(out_file, 'w')
    for line in inf:
        if normalize_probability:
            lang_prob = [0, 0, 0]
        else:
            lang_prob = [1, 1, 1]
        sentence = line
        
        # preprocess the sentence
        if is_case_folding:
            sentence = sentence.lower()
        if remove_punctuation:
            "".join(l for l in sentence if l not in string.punctuation)
        if add_padding:
            sentence = "!!!" + sentence + "$$$"
        
        for i in range(len(sentence) - n):
            ngram = sentence[i : i + n]
            # if there is a ngram don't exist in dictionary, then skip
            if ngram not in ngram_list:
                continue
            if normalize_probability:
                lang_prob[0] += ngram_prob[0][1][ngram]
                lang_prob[1] += ngram_prob[1][1][ngram]
                lang_prob[2] += ngram_prob[2][1][ngram]
            else:
                lang_prob[0] *= ngram_prob[0][1][ngram]
                lang_prob[1] *= ngram_prob[1][1][ngram]
                lang_prob[2] *= ngram_prob[2][1][ngram]
        
        if max(lang_prob) == lang_prob[0]:
            predict = 'malaysian'
        elif max(lang_prob) == lang_prob[1]:
            predict = 'indonesian'
        elif max(lang_prob) == lang_prob[2]:
            predict = 'tamil'
        if exist_other_language:
            if max(lang_prob) < threshold:
                predict = 'other'
        
        outf.write('{language} {sentence}'.format(language = predict, sentence = line))

def usage():
    print("usage: " + sys.argv[0] + " -b input-file-for-building-LM -t input-file-for-testing-LM -o output-file")

input_file_b = input_file_t = output_file = None
try:
    opts, args = getopt.getopt(sys.argv[1:], 'b:t:o:')
except getopt.GetoptError:
    usage()
    sys.exit(2)
for o, a in opts:
    if o == '-b':
        input_file_b = a
    elif o == '-t':
        input_file_t = a
    elif o == '-o':
        output_file = a
    else:
        assert False, "unhandled option"
if input_file_b == None or input_file_t == None or output_file == None:
    usage()
    sys.exit(2)

LM = build_LM(input_file_b)
test_LM(input_file_t, output_file, LM)
