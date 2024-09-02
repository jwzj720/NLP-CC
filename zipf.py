#!/usr/bin/env python3

import math
import os.path
from collections import Counter
from spacy.lang.en import English
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot

nlp = English(pipeline=[], max_length=5000000)

def H_approx(n):
    """
    Returns an approximate value of n-th harmonic number.
    http://en.wikipedia.org/wiki/Harmonic_number
    """
    # Euler-Mascheroni constant
    gamma = 0.57721566490153286060651209008240243104215933593992
    return gamma + math.log(n) + 0.5/n - 1./(12*n**2) + 1./(120*n**4)

def do_zipf_plot(counts, label="", total_tokens=None):
    """
    Let f be the relative frequency of a word 
    (e.g. if the occurs 1642 times out of 35652 tokens, then its relative frequency is 1642/35652 = 0.04606), 
    and let r be the rank of that word.
    To visualize the relationship between rank and frequency, 
    we will create a log-log plot of r (on the x-axis) versus f (on the y-axis). 
    For these plots, we will use the pyplot library, part of matplotlib.
    """

    fig = pyplot.figure()
    # Plot relative frequency vs rank of token in log-log scale
    pyplot.xlabel('Rank')
    pyplot.ylabel('Frequency')
    pyplot.title('Zipf\'s Law')
    # pyplot.grid(True)
    
    relative_frequency = {}
    total = sum(counts.values())
    
    for i in counts.elements():
        relative_frequency[i] = counts[i] / total
    
    sorted_counts = sorted(relative_frequency.items(), key=lambda x: x[1], reverse=True)
    freq_rank = {}
    
    for i, (word, freq) in enumerate(sorted_counts, 1):
        freq_rank[i] = freq
    
    # First Plot
    pyplot.loglog(freq_rank.keys(),freq_rank.values(), label = label)

    # ADD EXPECTED PLOT
    expected = {}
    print(total_tokens)
    k = total_tokens/H_approx(len(counts))
    
    for i, (word, freq) in enumerate(sorted_counts, 1):
        expected[i] = (k/i)/total_tokens

    pyplot.loglog(expected.keys(), expected.values(), label="Expected relative freq.", color="red")
    pyplot.legend()
    pyplot.savefig('zipf_{}.png'.format(label))
    pyplot.close()

def read_all(directory, extension=None):
    path = Path(directory)
    n = 0
    for file in path.iterdir():
        counter = Counter()
        if file.is_file() and (extension is None or file.suffix == extension):
            full_name = Path(file).name
            text = file.read_text()
            docs = nlp(text)
            n += len(docs)
            counter += Counter(doc.text.lower() for doc in docs)
    return (counter, n)
              

def read_one(fname):
    text = Path(fname).read_text()
    docs = nlp(text)
    n = len(docs)
    c = Counter(doc.text.lower() for doc in docs)
    #print(c)
    return (c, n)

def plot_all(directory):
    both = read_all(directory, ".txt")
    counts = both[0]
    tokens = both[1]
    do_zipf_plot(counts, os.path.basename(directory), tokens)

def plot_one(fname):
    both = read_one(fname)
    counts = both[0]
    tokens = both[1]
    title = os.path.splitext(os.path.basename(fname))[0]

    do_zipf_plot(counts, title, tokens)

def main():
    plot_one('/cs/cs159/data/gutenberg/milton-paradise.txt')
    plot_one('/cs/cs159/data/gutenberg/shakespeare-hamlet.txt')
    plot_one('random.txt')
    #plot_all('/cs/cs159/data/gutenberg')


if __name__ == "__main__":
    main()
