#!/usr/bin/env python3

import argparse
from lxml import etree
from collections import Counter
from html import unescape
from spacy.lang.en import English

nlp = English(pipeline=[], max_length=5000000)


def do_xml_parse(fp, tag):
    """ 
    Iteratively parses XML files
    """
    fp.seek(0)

    for (event, elem) in etree.iterparse(fp, tag=tag):
        yield elem
        elem.clear()

def get_examples(args, attribute, value):
    unigram_counter = Counter()
    bigram_counter = Counter()
    trigram_counter = Counter()
    
    for example in do_xml_parse(args.examples, tag="example"):
        if example.get(attribute) == value:
            for text_part in example.itertext():
                text = unescape(text_part)
                doc = nlp(text)
                
                unigrams = get_unigrams(doc)
                bigrams = get_bigrams(doc)
                trigrams = get_trigrams(doc)
                
                unigram_counter.update(unigrams)
                bigram_counter.update(bigrams)
                trigram_counter.update(trigrams)
    
    return unigram_counter, bigram_counter, trigram_counter


def get_unigrams(doc, do_lower=True): 
    # parse through SpaCY doc and return a list of unigrams
    unigrams = []
    for token in doc: 
        if do_lower:
            unigrams.append(token.text.lower())
        else:
            unigrams.append(token.text)
    return unigrams

def get_bigrams(doc, do_lower=True):
    bigrams = []
    tokens = [token.text.lower() if do_lower else token.text for token in doc]
    
    for i in range(len(tokens) - 1):
        bigram = (tokens[i], tokens[i + 1])
        bigrams.append(bigram)
    
    return bigrams

def get_trigrams(doc, do_lower=True):
    trigrams = []
    tokens = [token.text.lower() if do_lower else token.text for token in doc]
    
    for i in range(len(tokens) - 2):
        trigram = (tokens[i], tokens[i + 1], tokens[i + 2])
        trigrams.append(trigram)
    # print(trigrams)
    return trigrams

def compare(train, test, unique=False):

    train_zeros = 0
    count_total = 0

    if unique:
        test_unique = set(test.keys())
        train_unique = set(train.keys())
        train_zeros = len(test_unique - train_unique)
        count_total = len(test_unique)
    else:
        for token, count in test.items():
            if train[token] == 0:
                train_zeros += count
            count_total += count
        
    return (train_zeros, count_total)       
            
def do_experiment(args, attribute, train_value, test_value): 
    train_unigrams, train_bigrams, train_trigrams = get_examples(args, attribute, train_value) 
    test_unigrams, test_bigrams, test_trigrams = get_examples(args, attribute, test_value)

    table_header = "Results for {}, using {} as train and {} as test:"
    print(table_header.format(attribute, train_value, test_value))

    print("| Order   | Type/Token | Total | Zeros | % Zeros | ")
    print("| -------| -----------| ------| ------| --------| ")
    table_row = "| {ngram} | {typetoken} | {total} | {zeros} | {pct:.1%} | "

    for ngram, train, test in zip(
            ['Unigram', 'Bigram', 'Trigram'],
            [train_unigrams, train_bigrams, train_trigrams],
            [test_unigrams, test_bigrams, test_trigrams]):
        
        for do_types in (True, False):
            typetoken = "Type" if do_types else "Token"
            num_zeros, N = compare(train, test, do_types)
            print(table_row.format(ngram=ngram, typetoken=typetoken, 
                  total=N, zeros=num_zeros, pct=num_zeros/N))
    print()

def main(args):

    do_experiment(args, "condescension", "true", "false")
    do_experiment(args, "condescension", "false", "true")
    #print("Randomized Chunk Experiment:")
    do_experiment(args, "randomchunk", "a", "b")
    do_experiment(args, "randomchunk", "b", "a")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 'rb' means "read as bytes", which means that it doesn't assume
    # the data is UTF-8 text when it's read in.
    parser.add_argument("--examples", "-a",
                        type=argparse.FileType('rb'),
                        help="Content of examples")

    args = parser.parse_args()

    main(args)
