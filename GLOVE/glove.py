#!/usr/bin/env python3

import argparse
import numpy

def load_text_vectors(fp):
    rows = 0
    columns = 0

    for line in fp:
        rows += 1
        if columns == 0:
            columns = len(line.split()) - 1

    vectors = numpy.zeros((rows, columns))
    
    words = []

    fp.seek(0)

    for i, line in enumerate(fp):
        l = line.split()
        word = l[0]
        vector = [float(x) for x in l[1:]]
        words.append(word)
        vectors[i] = vector
    
    return (words, vectors)

def save_glove_vectors(word_list, vectors, fp):
    numpy.savez(fp, vectors=vectors, words=word_list)
    fp.close()

def load_glove_vectors(fp):
    data = numpy.load(fp, allow_pickle=True)
    vectors = data['vectors']
    words = list(data['words'])
    return (vectors, words)

def get_vec(word, wordlist, array):
    word_index = wordlist.index(word)
    word_vector = array[word_index]
    # print(word_index + ", " + word_vector)
    return word_vector

def main(args): 
    words, vectors = load_text_vectors(args.GloVeFILE)
    save_glove_vectors(words, vectors, args.npyFILE)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("GloVeFILE",
                        type=argparse.FileType('r'),
                        help="a GloVe text file to read from")
    parser.add_argument("npyFILE",
                        type=argparse.FileType('wb'),
                        help='an .npy file to write the saved numpy data to')

    args = parser.parse_args()
    main(args)
