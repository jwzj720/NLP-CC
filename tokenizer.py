#!/usr/bin/env python3

from collections import Counter
import os
import re
from pathlib import Path

def get_words(s):
    l = str.split(s)
    return l
    

def count_words(list_of_words, to_lower=False): 
    if to_lower:
        list_of_words = [w.lower() for w in list_of_words]
    return Counter(list_of_words)

def words_by_frequency(list_of_words, n=None, to_lower=False):
    if to_lower:
        list_of_words = [w.lower() for w in list_of_words]
    
    return Counter(list_of_words).most_common(n)

def is_space(s):
   #print(s)
   return not s.strip() == ""
   
def tokenize(s, to_lower=True):
   split_string = re.split(r'(\W)', s)
   if to_lower:
      split_string = [word.lower() for word in split_string]
    
   split_string = list(filter(is_space, split_string))
    
   return split_string


def filter_nonwords(list_of_tokens):
    return list(filter(str.isalpha, list_of_tokens))

def main(): 
    gutenburg_path = Path("/cs/cs159/data/gutenberg/")
    for file in gutenburg_path.iterdir():
        full_name = Path(file).name
        # print(full_name)
        if file.is_file() and full_name.endswith(".txt"):
            print(full_name + ": ")
            print(words_by_frequency(filter_nonwords(tokenize(file.read_text())), 5, to_lower=True))
              
  
if __name__ == '__main__':
    main()
