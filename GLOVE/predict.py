#!/usr/bin/env python3

from glove import *
from similarity import *
from visualize import *
import argparse
import numpy as np
import random
import sys
from sklearn.model_selection import train_test_split

def average_difference(first_vectors, second_vectors):
    return np.mean(first_vectors-second_vectors, axis=0)


def do_experiment(args):
    
    vectors, word_list = glove.load_glove_vectors(args.npyFILE)
    
    relationships = read_relations(args.relationsFILE)

    random.shuffle(relationships)

    split_point = int(0.8 * len(relationships))

    train_relations = relationships[:split_point]

    test_relations = relationships[split_point:]
    
    train_first_vectors, train_second_vectors, train_filtered_relations = extract_words(vectors, word_list, train_relations)
    
    train_average_difference = average_difference(train_first_vectors, train_second_vectors)
    print(train_average_difference)
    mean_reciprocal_rank = 0
    first_most_similar = 0
    first_top_ten = 0
    
    for first_word, second_word in test_relations:
        if second_word in word_list:
            if args.average:
                #print("average\n")
                second_vector = vectors[word_list.index(second_word)] 
                second_vector += train_average_difference
            else:
                second_vector = vectors[word_list.index(second_word)]
            closest_words = closest_vectors(second_vector, word_list, vectors, 101)[1:]
            closest_words_list = [word for _, word in closest_words]
            #print(closest_words_list)
            if first_word in closest_words_list:
                rank = closest_words_list.index(first_word) + 1
                if rank == 1:
                    first_most_similar += 1
                if rank <= 10:
                    first_top_ten += 1
                mean_reciprocal_rank += 1.0 / rank
            else:
                mean_reciprocal_rank += 0
                
    print(f"Top word compared to test_relations: {(first_most_similar / len(test_relations))*100:.4f}%")
    print(f"Top 10 compared to test_relations: {first_top_ten / len(test_relations)*100:.4f}%")
    print(f"Mean Reciprocal Rank: {mean_reciprocal_rank / len(test_relations):.4f}")

def main(args):
    do_experiment(args)
    #vectors, word_list = load_glove_vectors(open('data/glove.6B.50d.npy', 'rb'))
    #relations = read_relations(open('/cs/cs159/data/glove/relations/capitals.txt', 'r'))
    #first_vectors, second_vectors, filtered_relations = extract_words(vectors, word_list, relations)
    #print(average_difference(first_vectors, second_vectors))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run expiriement")
    parser.add_argument("npyFILE",
                        type=argparse.FileType('rb'),
                        help='an .npy file to read the saved numpy data from')
    parser.add_argument("relationsFILE",
                        type=argparse.FileType('r'),
                        help='a file containing pairs of relations')
    parser.add_argument('-a', '--average', action='store_true')

    args = parser.parse_args()
   
    main(args)
