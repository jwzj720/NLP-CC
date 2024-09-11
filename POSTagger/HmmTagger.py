#!/usr/bin/env python3
import argparse
from collections import defaultdict, Counter
import os
import pickle
import sys

from numpy import argmax, zeros, array, float32, ones, zeros, log
import spacy

import read_tags


class HMMTagger():

    def __init__(self, nlp, alpha=0.1, args=None):
        if args and args.universal:
            self.tags = ["<<START>>",] + read_tags.universal_tag_set
        else:
            self.tags = ["<<START>>",] + read_tags.ptb_tag_set
        self.vocab = ["<<OOV>>",]
        self.alpha = alpha
        if args: 
            self.do_universal = args.universal 
            self.vocabsize = args.vocabsize
        else: 
            self.do_universal = False
            self.vocabsize = None 

    def clean_token(self, text):
        """Convert each token to lower-case text or a special OOV token
        for out of vocabulary words"""
        return text.lower() if text.lower() in self.vocab else "<<OOV>>"

    def word_to_index(self, w):
        """Given a token, find its index in the vocabulary."""
        return self.vocab.index(self.clean_token(w))

    def tag_to_index(self, t):
        """Given a tag, find its index in the tag list."""
        return self.tags.index(t)

    def update_vocab(self, train_dir):
        """Given a directory of files, populate the vocabulary that corresponds
        to all tokens in the files in the directory."""
        token_counter = Counter()
        for words, _ in read_tags.parse_dir(train_dir): 
            token_counter.update([w.lower() for w in words])

        if self.vocabsize:
            most_common_words = [word for word, _ in token_counter.most_common(self.vocabsize)]
        else:
            most_common_words = list(token_counter.keys())
        

        self.vocab += most_common_words
    
    def normalize_probabilities(self):
        """Normalize the tag-word and tag-tag probability matrices
        in log space."""
        self.tag_word_probs = self.normalize(self.tag_word)
        self.tag_tag_probs = self.normalize(self.tag_tag)

    def __call__(self, tokens):
        """If invoked as a function call, predicts tags for a list
        of tokens given the trained model."""
        self.predict(tokens)
        
    ## BEGIN DOCUMENTATION HERE ###

    def get_start_costs(self):
        """
        Returns the list of probabilities it is the first tag

        This method returns the list of probabilities that a tag is the first tag
        in a sentence.

        Returns:
            numpy.ndarray: list of probabilities for each tag being the first tag
        """
        return self.tag_tag_probs[self.tag_to_index("<<START>>"),:]

    def get_token_costs(self, token):
        """
        Returns a list of probabilities of token being each tag

        This method takes a token and returns a list of probabilities
        for the token being each tag.

        Args:
            token (spacy.Token): token to be tagged
        
        Returns:
            numpy.ndarray: list of probabilities for the token being each tag
        """
        return self.tag_word_probs[:,self.word_to_index(token.text)]
    
    def normalize(self, m):
        """
        Normalizes the matrix in log space

        This method takes a matrix and normalizes it to log probabilities.
        It computes the log of each element, then the log sum of the row, 
        and then transpososes it back. 

        Args:
            m (numpy.ndarray): matrix to be normalized
        
        Returns:
            numpy.ndarray: normalized matrix in log space
        """
        return (log(m).transpose() - log(m.sum(axis=1))).transpose()

    def do_train_sent(self, words, tags): 
        """
        Trains the tagger on one sentance. 

        This method trains the HMM tagger on a single sentence,
        updating the emission and tranisiton counts.

        Args:
            words (list): list of words in the sentence
            tags (list): list of tags in the sentence
        
        Returns:
            None: function modifies the tag_word and tag_tag matrices in place

        """
        prev_tag = self.tag_to_index("<<START>>")
        for word, tag in zip(words, tags):
            t_i = self.tag_to_index(tag)
            w_i = self.word_to_index(word)
            self.tag_word[t_i][w_i] += 1
            self.tag_tag[prev_tag][t_i] += 1
            prev_tag = t_i

    def train(self, train_dir):
        """
        Trains the HMM tagger on the data from a specified directory

        This method updates the vocabulary, initializes the transition 
        and emission probability matrices, and then trains the model.

        Args:
            train_dir (str): directory containing the training data
        
        Returns:
            None: function modifies the tag_word and tag_tag matrices in place
        """
        self.update_vocab(train_dir)

        # which of these gives transition probabilities,
        # and which gives emission probabilities?
        # self.tag_word gives emission probabilities (P(word|tag))
        # self.tag_tag gives transition probabilities (P(tag|previous_tag))
        self.tag_word  = ones((len(self.tags), len(self.vocab))) * self.alpha
        self.tag_tag =  ones((len(self.tags), len(self.tags))) * self.alpha

        for words, tags in read_tags.parse_dir(train_dir, self.do_universal):
            self.do_train_sent(words, tags)
        self.normalize_probabilities()

    def predict(self, tokens):
        """
        Predict the sequence of tags for the tokens using the Viterbi algorithm.

        This method uses the Viterbi algorithm to find the most likely sequence of tags. 
        It fills in the cost table and the backtrace table, then calls backtrace to 
        determine the best sequence of tags.

        Args:
            tokens (list): list of tokens to be tagged.
        
        Returns:
            None: function modifies the tokens in place
        """
        # Build DP table, which should be |sent| x |tags|
        cost_table = zeros((len(tokens), len(self.tags)), float32)
        bt_table   = zeros((len(tokens), len(self.tags)), int)

        for token_i, token in enumerate(tokens):
            token_costs = self.get_token_costs(token)
            if token_i == 0: 
                cost_table[token_i, :] = self.get_start_costs() + token_costs # Adding because logs: log(A) + log(B) = log(A*B)
                bt_table[token_i, :] = -1
            else:
                costs = self.tag_tag_probs.copy()

                # TODO: Fill in the actual costs matrix to compute
                # the sum of the log probability from the last state,
                # the transition log probability,
                # and the emission log probability
                costs += token_costs
                # multiply eacgh row of table 8.12 (transition probs)
                # by the column of table 8.13 (emission probs)
                costs = costs.transpose() + cost_table[token_i-1,:]
                # Transpose it back to the original shape
                costs = costs.transpose()

                

                # print(cost.shape, token_costs.shape, cost_table[token-1,:].shape)
                cost_table[token_i, :] = costs.max(axis=0)
                bt_table[token_i,:] = costs.argmax(axis=0)

        # Find the highest-probability tag for last word
        best_last_tag = argmax(cost_table[token_i, :])

        # Trace back through the breadcrumb table
        self.backtrace(bt_table, tokens, best_last_tag)

    def backtrace(self, bt_table, tokens, best_last_tag):
        """
        Perform backtracing to determine the best sequence of tags for the given tokens.

        This method uses the bt_table table to find the likely sequence of tags
        for the tokens, starting from the best_last_tag and working backwards.

        Args:
            bt_table (numpy.ndarray): 2D array where each entry is the best previous tag
                                    for a given token and current tag.
            tokens (list): list of tokens to be tagged.
            best_last_tag (int): index of the best tag for the last token.

        Returns:
            None: function modifies the tokens in place 
        """
        current_row = len(tokens)-1
        for t in list(tokens)[::-1]:
            t.tag_ = self.tags[best_last_tag]
            best_last_tag = bt_table[current_row, best_last_tag]
            current_row -= 1


def main(args):
    nlp = spacy.load("en_core_web_sm")
    tagger = HMMTagger(nlp, alpha=args.alpha, args=args)
    tagger.train(args.dir)
    pickle.dump(tagger, args.output)


if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Train (and save) hmm models for POS tagging')
    parser.add_argument("--dir", "-d", metavar="DIR", required=True,
                        help="Read training data from DIR")
    parser.add_argument("--output", "-o", metavar="FILE", 
                        type=argparse.FileType('wb'), required=True,
                        help="Save output to FILE")
    parser.add_argument("--alpha", "-a", default=0.1, 
                        help="Alpha value for add-alpha smoothing")
    parser.add_argument("--universal", "-u", action="store_true", help="If set, uses universal tags instead of PTB tag set")
    parser.add_argument("--vocabsize", "-v", type=int, default=None, 
                        help="Specify the maximum vocabulary size. If not set, keeps all words.")

    args = parser.parse_args()
    
    main(args)
