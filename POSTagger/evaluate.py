#!/usr/bin/env python3
import argparse
import os
import pickle
import spacy
import sys
import numpy

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from HmmTagger import HMMTagger
from read_tags import *


def main(args): 
    nlp = spacy.load("en_core_web_sm")
   
    # Load the trained HMM model from the pickle file
    tagger = pickle.load(args.hmm)
    
    doc_lengths = []
    accuracies = []
    
    for fname in get_files(args.dir): 
        with open(fname) as fp:
            all_words = []
            all_tags = []
            
            for words, tags in parse_file(fp, do_universal=args.universal):
                all_words.extend(words)
                all_tags.extend(tags)
        
        doc = nlp.tokenizer.tokens_from_list(list(all_words))
        tagger(doc)  

        if args.universal and not tagger.do_universal: 
            for token in doc:
                token.tag_ = ptb_to_universal.get(token.tag_, token.tag_)
        
        correct_tags = 0 
        total_tags = 0    

        for spacy_token, ref_tag in zip(doc, all_tags):
            if spacy_token.tag_ == ref_tag:
                correct_tags += 1  
            total_tags += 1  

        total_right = correct_tags  
        total_size = total_tags     
        
        doc_lengths.append(total_size)             
        accuracies.append(total_right / total_size * 100)  

    if accuracies:
        overall_accuracy = sum(accuracies) / len(accuracies)
        print(f"Overall Accuracy: {overall_accuracy:.2f}%")
    
    if args.output:
        plt.scatter(doc_lengths, accuracies)
        z = numpy.polyfit(doc_lengths, accuracies, 1)
        p = numpy.poly1d(z)
        plt.plot(doc_lengths, p(doc_lengths), "r--", label='Best Fit Line')
        plt.xlabel('Document Size (tokens)')
        plt.ylabel('Accuracy (%)')
        plt.title('Document Size vs. POS Tagging Accuracy')
        plt.savefig(args.output)  
        print(f"Scatter plot saved to {args.output}")

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='POS Tag, then evaluate')
    parser.add_argument("--dir", "-d", metavar="DIR", required=True)
    parser.add_argument("--hmm", metavar="FILE", 
                        type=argparse.FileType('rb'), required=True)
    parser.add_argument("--universal", "-u", action="store_true")
    parser.add_argument("--output", "-o", metavar="FILE", default=None)
    args = parser.parse_args()
    main(args)