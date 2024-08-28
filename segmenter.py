#!/usr/bin/env python3

import sys
import argparse
from tokenizer import tokenize
import os

def my_best_segmenter(token_list): 
    """ TODO: Replace this with an improved sentence segmenter. """
    pass

def baseline_segmenter(token_list):
    all_sentences = []
    this_sentence = []
    for token in token_list:
        this_sentence.append(token)
        if token in ['.', ':', ';', '!', '?']:
            all_sentences.append(this_sentence)
            this_sentence = []
    return all_sentences

def write_sentence_boundaries(sentence_list, out):
    """ TODO: Write out the token numbers of the sentence boundaries. """
    for sentence in sentence_list:
        #out.write(str(sentence) + "\n")
        out.write(str(sentence.index(sentence[-1])) + "\n")

def main(args):
    #print(len(tokenize(args.textfile.read())))
    with open('out.txt', 'w') as out_file:
        print("Baseline segmenter:")
        sentences = baseline_segmenter(tokenize(args.textfile.read(), to_lower=True))
        write_sentence_boundaries(sentences, out_file)

"""You may have opened a file using something like
        with open(file_name) as file_pointer:
or
        file_pointer = open(file_name, 'w')
where file_pointer would refer to a pointer to some place in memory with a
file.  As you read or write information, the location of that pointer in memory
will change. This is different from how we often think of opening a file, as
the program won't necessarily know the whole contents of the file when you
first open it. In situations where we have huge files with many lines, we can
use this to loop through line by line without having to keep everything in
memory at once.

In order to get lines from the file, we can say
        file_pointer.read_line()
or iterate through lines like
        for line in file_pointer:

Throughout this class, we'll have ArgumentParsers helping to pass in files.  If
you tell an ArgumentParser that a passed-in string is a path to a filename, it
can open the file for you: for instance, --textfile below will allow us to
access the file pointer (not just the filename) as the variable args.textfile.
To do this, you also need to specify the mode to open the file: here, the
hypothesis file is open under write mode 'w', unlike the text file, which is
open in read mode 'r'. If you use this functionality, you shouldn't ever call
the open command yourself: you should tread the passed object the same way as
you treated file_pointer in the snippets above.
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sentence Segmenter for NLP Lab")
    parser.add_argument('--textfile', "-t", metavar="FILE", type=argparse.FileType('r'),
                        required=True, help="Unlabeled text is in FILE.")
    parser.add_argument("--hypothesis_file", "-y", metavar="FILE", type=argparse.FileType('w'),
                        required=False, default=sys.stdout,
                        help="Write hypothesized boundaries to FILE")

    args = parser.parse_args()
    main(args)
