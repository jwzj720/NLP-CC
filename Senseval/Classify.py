#!/usr/bin/env python3

import argparse
import math
import numpy as np
from Lexelt import get_data, get_key
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score

def main(args): 

    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_features': ['sqrt', 'log2', None],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'bootstrap': [True, False]
    }

    # Load training and test data
    train_data = get_data(args.traindata)
    get_key(args.trainkey, train_data)
    test_data = get_data(args.testdata)
    get_key(args.testkey, test_data)

    correct = 0
    total = 0


    for key, lexelt in train_data.items(): 
        test_lexelt = test_data[key]

        # Load the training and test features
        feature_labels, X_train = lexelt.get_features()
        _, X_test = test_lexelt.get_features(feature_labels)

        targets, y_train = lexelt.get_targets()
        _, y_test = test_lexelt.get_targets(targets)
        transformer = TfidfTransformer(smooth_idf=False)
        transformer.fit(X_train)
    
        X_train = transformer.transform(X_train).toarray()
        X_test = transformer.transform(X_test).toarray()
        rf = RandomForestClassifier(random_state=0, bootstrap = False)
        
        #stratified_kfold = StratifiedKFold(n_splits=3)

        #random_search = RandomizedSearchCV(
        #    estimator=rf, 
        #    param_distributions=param_grid, 
        #    n_iter=20,  
        #   cv=stratified_kfold, 
        #    verbose=1, 
        #    random_state=0, 
        #    n_jobs=-1  
        #    )

        rf.fit(X_train, y_train)

        # Get the best model and predict test data
        #best_rf = rf.best_estimator_
        y_pred = rf.predict(X_test) 

        # Ensure correct prediction length
        assert len(y_pred) == len(y_test), "Mismatch in prediction length."

        # Count correct predictions
        correct += sum(y_pred == y_test)
        total += len(y_pred)
        print(key)

        importances = sorted(list(zip(feature_labels, rf.feature_importances_)), key=lambda x: x[1], reverse=True)[0:10]

        print(importances, '\n')
    
    # Print the overall accuracy
    print(f"Classifier score: {correct / total:.2%}")

    
    
if __name__ == "__main__": 

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
