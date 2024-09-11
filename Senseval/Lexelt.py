import argparse
from bs4 import BeautifulSoup as bs
import os
from collections import Counter, defaultdict
import numpy
# import array_to_latex as a2l
##
 # Harvey Mudd College, CS159
 # Swarthmore College, CS65
 # Copyright (c) 2018, 2020 Harvey Mudd College Computer Science Department, Claremont, CA
 # Copyright (c) 2018 Swarthmore College Computer Science Department, Swarthmore, PA
##

class LexElt(): 
    def __init__(self, lexelt, skip_stopwords=False, do_casefolding=False, do_lemmas=False): 
        self.key = lexelt['item']
        self._instances = dict([(i['id'], LexEltInstance(i))
                                for i in lexelt.find_all('instance')])
        self.features = defaultdict(Counter)
        self.skip_stopwords = skip_stopwords
        self.do_casefolding = do_casefolding
        self.do_lemmas = do_lemmas

    def add_answer(self, instance_id, answers): 
        self._instances[instance_id].add_answer(answers)
    def instances(self): 
        return self._instances.values()
    def get(self, instance_id): 
        return self._instances[instance_id]
    def keys(self): 
        return self._instances.keys()
    def get_instance(self, instance_id): 
        return self._instances[instance_id]

    def pos(self): 
        ## TODO: Implement this! 
        return self.key.split('.')[-1]

    def num_headwords(self): 
        ## TODO: Implement this! 
        return [len(instance.heads) for instance in self.instances()]

    def num_answers(self): 
        ## TODO: Implement this! 
        return [len(instance.answers) if instance.answers else 0 for instance in self.instances()]

    def get_all_senses(self): 
        ## TODO: Implement this! 
        all_senses = []
        for instance in self.instances():
            if instance.answers:
                all_senses.extend([ans for ans in instance.answers if ans != 'U'])
        return all_senses

    def count_unique_senses(self): 
        ## TODO: Implement this! 
        return len(set(self.get_all_senses()))

    def most_frequent_sense(self): 
        ## TODO: Implement this! 
        return Counter(self.get_all_senses()).most_common(1)[0][0]

    def get_features(self, feature_names=None):
        ## TODO: Implement this! 
        all_features = set()
        if feature_names is None:
            for i in self._instances.values():
                i.make_features()
                all_features.update(i.features.keys())
            feature_names = sorted(all_features) # NEED TO SORT 

        num_instances = len(self._instances)
        num_features = len(feature_names)
        feature_array = numpy.zeros((num_instances, num_features))

        for i in range(len(self._instances)):
            instance = list(self._instances.values())[i]
            feature_array[i, :] = instance.to_vector(feature_names)

        return feature_names, feature_array
        

    def get_targets(self, labels=None): 
        unique_answers = set()
        instance_first_answers = []
        
        for i in self._instances.values():
            first_answer = i.answers[0]
            unique_answers.add(first_answer)
            instance_first_answers.append(first_answer)
        
        if labels is None:
            sorted_answers = sorted(unique_answers)
        else:
            sorted_answers = labels 

        answer_to_int = {answer: index for index, answer in enumerate(sorted_answers)}
        y = numpy.array([answer_to_int.get(answer, -1) for answer in instance_first_answers])
        return sorted_answers, y

class LexEltInstance(): 

    def __init__(self, instance):
        self.id = instance['id']
        self.words = []
        self.heads = []
        self.doc = None
        self.answers = None
        self.features = None

        for c in instance.context.contents: 
            self.add_context(c)

    def add_context(self, c): 
        if hasattr(c, 'contents'): # Head word
            text = c.contents[0]
            self.heads.append(len(self.words))
        else: # Not a head word
            text = c.string.strip()
        self.words.extend(text.split())

    def add_answer(self, answers): 
        self.answers = answers
    
    def has_answer(self, a): 
        return a in self.answers

    ## Start functions that students should write. 
    def to_vector(self, feature_list): 
        ## TODO: Implement this! 
        if self.features is None:
            self.make_features()
        feature_vector = numpy.array([self.features.get(feature, 0) for feature in feature_list])

        return feature_vector
    
    def bow_features(self): 
        ## TODO: Implement this! 
        return Counter(self.words)

    def colloc_features(self): 
        ## TODO: Implement this! 
        collocations = Counter()
        bigrams = self.bigrams()
        trigrams = self.trigrams()

        bigram_features = ['_'.join(bigram) for bigram in bigrams]
        trigram_features = ['_'.join(trigram) for trigram in trigrams]

        collocations.update(bigram_features)
        collocations.update(trigram_features)

        return collocations

    def make_features(self): 
        ## TODO: Implement this! 
        bow = self.bow_features()
        colloc = self.colloc_features()

        # Combine both Counters
        self.features = bow + colloc

    def get_feature_names(self): 
        ## TODO: Implement this! 
        if self.features is None:
            self.make_features()

        return self.features.keys()
      
    # helpers for colloc_features
    def bigrams(self): 
        N = len(self.words)
        return [(self.words[i-1], self.words[i]) 
                for i in self.heads 
                if i > 0] +\
               [(self.words[i], self.words[i+1])
                for i in self.heads
                if i < N-1]
    
    def trigrams(self): 
        N = len(self.words)
        return [(self.words[i-2], self.words[i-1], self.words[i]) 
                for i in self.heads
                if 2 <= i] +\
               [(self.words[i-1], self.words[i], self.words[i+1])
                for i in self.heads
                if 1 <= i < N-1] +\
               [(self.words[i], self.words[i+1], self.words[i+2])
                for i in self.heads
                if i < N-2]


def get_data(fp):
    """
    Input: input file pointer (training or test)

    Return: a dictionary mapping "lexelts" to LexElt objects. 
    Each LexElt object stores its own instances.
    """

    soup = bs('<doc>{}</doc>'.format(fp.read()),"xml")
    
    return dict([(lexelt['item'], LexElt(lexelt)) 
                 for lexelt in soup.findAll('lexelt')])


def get_key(fp, data):
    """ read the answer key """
    for line in fp:
        fields = line.split()
        target = fields[0]
        instance_ID = fields[1]
        answers = fields[2:]
        data[target].add_answer(instance_ID, answers)


def main(args):
    train_data = get_data(args.traindata)
    get_key(args.trainkey, train_data)

    test_data = get_data(args.testdata)
    get_key(args.testkey, test_data)
    
    print(len(train_data))
    types = set()
    breakdown = defaultdict(int)
    for i in train_data:
        if '.' in i:
            suffix = i.split('.')[-1]
            types.add(suffix)
            breakdown[suffix] += 1

    total_count = len(train_data)
    percentages = {suffix: (count / total_count) * 100 for suffix, count in breakdown.items()}

    print(types)
    print(percentages)


    lexelt = train_data['organization.n']
    print(len(lexelt.instances()))

    headword_counts = Counter()
    answer_counts = Counter()

    for lexelt in train_data.values():
        headword_counts.update(lexelt.num_headwords())
        answer_counts.update(lexelt.num_answers())
    headword_table = numpy.array([[key, value] for key, value in headword_counts.items()])
    answer_table = numpy.array([[key, value] for key, value in answer_counts.items()])
    
    print(headword_counts)
    print(answer_counts)

    lexelt = train_data['organization.n']
    print(len(lexelt.get_all_senses()))

    total_random_correct = 0
    total_instances = 0

    for lexelt in train_data.values():
        num_instances = len(lexelt._instances)
        num_senses = lexelt.count_unique_senses()  
        
        random_correct = num_instances / num_senses
        total_random_correct += random_correct
        
        total_instances += num_instances


    random_accuracy = total_random_correct / total_instances * 100
    print(f"Random guessing accuracy: {random_accuracy:.2f}%")

    total_mfs_correct = 0
    total_instances = 0

    for lexelt in train_data.values():
        most_frequent_sense = lexelt.most_frequent_sense()

        for instance in lexelt.instances():
            if instance.answers:
                first_answer = instance.answers[0]  
                if first_answer == most_frequent_sense:
                    total_mfs_correct += 1
            total_instances += 1

    mfs_accuracy = total_mfs_correct / total_instances * 100
    print(f"MFS accuracy: {mfs_accuracy:.2f}%")
    ### TEST DATA EXPLORATION
    lexelt = test_data['activate.v']
    print(len(lexelt.instances()))

    train_lexelt = train_data['organization.n']
    train_senses = set(train_lexelt.get_all_senses())

    test_lexelt = test_data['organization.n']
    test_senses = set(test_lexelt.get_all_senses())

    senses_in_train_not_test = train_senses - test_senses
    senses_in_test_not_train = test_senses - train_senses

    print(f"Senses in training but not in test: {len(senses_in_train_not_test)}")
    print(f"Senses in test but not in training: {len(senses_in_test_not_train)}")

    total_mfs_correct = 0
    total_test_instances = 0
    mfs = {}

    for key, lexelt in train_data.items():
        most_frequent_sense = lexelt.most_frequent_sense()
        mfs[key] = most_frequent_sense

    for key, test_lexelt in test_data.items():
        if key in mfs:
            mfs_val = mfs[key]
        for instance in test_lexelt.instances():
            first_answer = instance.answers[0]
            if first_answer == mfs_val:
                total_mfs_correct += 1
            total_test_instances += 1

    mfs_accuracy = total_mfs_correct / total_test_instances * 100
    print(f"MFS accuracy on test data: {mfs_accuracy:.2f}%")
   
    ### CHECK POINTS
    ## Test for part 1
    #print(train_data['activate.v'].pos())  
    #print(train_data['activate.v'].num_headwords())
    #print(train_data['activate.v'].num_answers())
    #print(train_data['activate.v'].get_all_senses())
    #print(train_data['activate.v'].count_unique_senses())
    #print(train_data['activate.v'].most_frequent_sense())
    ## TODO: Put code here to answer the Lexelt questions.
    #this_instance = lexelt.get('activate.n.bnc.00044852')
    #this_instance.make_features()
    #print(this_instance.features)
    #print(this_instance.get_feature_names())
    #this_instance.to_vector(['and', 'types', 'system_are_activated'])
    #feature_names, X_train = lexelt.get_features()
    #print(feature_names)
    #print(X_train)
    #print("1) There are {} different lexelts.".format(len(train_data)))
    #print(X_train.shape)
    #answer_labels, y_train = lexelt.get_targets()
    #print(answer_labels)
    #print(y_train)
    #test_fp = open('/cs/cs159/data/senseval3/test/EnglishLS.test', 'r')
    #test_data = get_data(test_fp)
    #testkey_fp = open('/cs/cs159/data/senseval3/test/EnglishLS.test.key', 'r')
    #get_key(testkey_fp, test_data)
    #test_lexelt = test_data['activate.v']
    #_, X_test = test_lexelt.get_features(feature_names)
    #print(X_test)
    #_, y_test = test_lexelt.get_targets(answer_labels)
    #print(y_test)
if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--traindata",
                        type=argparse.FileType('r'),
                        default='/cs/cs159/data/senseval3/train/EnglishLS.train',
                        help='xml file containing training examples')
    parser.add_argument("--testdata",
                        type=argparse.FileType('r'),
                        default='/cs/cs159/data/senseval3/test/EnglishLS.test',
                        help='xml file containing test examples') 
    parser.add_argument("--trainkey",
                        type=argparse.FileType('r'),
                        default='/cs/cs159/data/senseval3/train/EnglishLS.train.key',
                        help='file containing training labels')  
    parser.add_argument("--testkey",
                        type=argparse.FileType('r'),
                        default='/cs/cs159/data/senseval3/test/EnglishLS.test.key',
                        help='file containing test labels')                     

    args = parser.parse_args()
    main(args)
