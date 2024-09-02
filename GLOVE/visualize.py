#!/usr/bin/env python3

import glove
from sklearn.decomposition import PCA  # put this at the top of your program
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from similarity import *
import argparse
from itertools import chain
import numpy as np


def read_relations(fp):
    relations = []
    first = True
    for line in fp.readlines():
        if not first:
            relations.append(tuple(line.strip().split()))
        else:
            first = False
    return relations

def perform_pca(array, n_components=2):

    # For the purposes of this lab, n_components will always be 2.

    pca = PCA(n_components=n_components)

    pc = pca.fit_transform(array)

    return pc


def extract_words(vectors, word_list, relations):

    pairs  = [(word1, word2) for word1, word2 in relations if word1 in word_list and word2 in word_list]
    vectors1 = np.array([vectors[word_list.index(word1)] for word1, word2 in pairs])
    vectors2 = np.array([vectors[word_list.index(word2)] for word1, word2 in pairs])

    return vectors1, vectors2, pairs    

def plot_relations(pca_first, pca_second, pca_relations, filename='plot.png'):
    fig = plt.figure(figsize = (8,8))

    ax = fig.add_subplot(1,1,1)

    ax.scatter(pca_first[:,0], pca_first[:,1], c='r', s=50)

    ax.scatter(pca_second[:,0], pca_second[:,1], c='b', s=50)

    for i in range(len(pca_first)):

        (x,y) = pca_first[i]

        plt.annotate(pca_relations[i][0], xy=(x,y), color="black")

        (x,y) = pca_second[i]

        plt.annotate(pca_relations[i][1], xy=(x,y), color="black")

    for i in range(len(pca_first)):

        (x1,y1) = pca_first[i]

        (x2,y2) = pca_second[i]

        ax.plot((x1, x2), (y1, y2), linewidth=1, color="lightgray")
    
    plt.savefig(filename)

def main(args):
    vectors, word_list = glove.load_glove_vectors(args.npyFILE)
    relations = read_relations(args.relationsFILE)
    #print(extract_words(vectors, word_list, relations))
    first_vectors, second_vectors, pairs = extract_words(vectors, word_list, relations)
    array = numpy.vstack((first_vectors, second_vectors))
    #print(pairs)
    pca_vectors = perform_pca(array, 2)
    #print(pca_vectors)
    pca_first = pca_vectors[:len(pairs)]
    pca_second = pca_vectors[len(pairs):]
    plot_relations(pca_first, pca_second, pairs, args.plot)

    
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find the n closest words to " +\
                      "a given word (if specified) or to all of the " +\
                       "words in a text file (if specified). If " +\
                       "neither is specified, compute nothing.")
    parser.add_argument("npyFILE",
                        type=argparse.FileType('rb'),
                        help='an .npy file to read the saved numpy data from')
    parser.add_argument("relationsFILE",
                        type=argparse.FileType('r'),
                        help='a file containing pairs of relations')
    parser.add_argument("--plot", "-p", default="plot.png", help="Name of file to write plot to.")

    args = parser.parse_args()
    main(args)
