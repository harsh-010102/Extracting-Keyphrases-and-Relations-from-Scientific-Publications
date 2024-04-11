from __future__ import division
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import sys
import PositionRank
import evaluation
import process_data
import argparse
import os
from os.path import isfile, join
from nltk.stem.porter import PorterStemmer
porter_stemmer = PorterStemmer()
import nltk
nltk.download('averaged_perceptron_tagger')


def process(input_data, output_data, topK, window):    
    files = []
    for f in os.listdir(input_data):
        if isfile(join(input_data, f)):
            files.append(join(input_data, f))
    Rprec = 0.0
    bpref = 0.0
    docs = 0
    P, R, F1 = [0] * topK, [0] * topK, [0] * topK
    # print(files)


    for file_name in files:
        # print(args.input_data + filename) 
        _, gold_filename = os.path.split(file_name)

        text_files = process_data.read_input_file(file_name)
        key_phrases = process_data.read_gold_file(output_data + gold_filename[:-3] +'key')
        # break
        # print(type(key_phrases))
        if text_files and key_phrases:
            gold_stemmed = []
            for keyphrase in key_phrases:
                keyphrase = [porter_stemmer.stem(w) for w in keyphrase.lower().split()]
                gold_stemmed.append(' '.join(keyphrase))
            # count the document
            docs += 1
            # initialize the Position Rank
            phrase_type = 'n_grams'
            system = PositionRank.PositionRank(text_files, window, phrase_type)
            # get_doc_words -> to make words structure which can be a potential keyphrase 
            system.get_doc_words()

            system.candidate_selection()

            system.candidate_scoring(window, update_scoring_method=False)
            # print(len(system.get_best_k(topK)))
            currentP, currentR, currentF1 = evaluation.PRF_range(system.get_best_k(topK), gold_stemmed, cut_off=topK)
            # print(system.get_best_k(args.topK), gold_stemmed)
            Rprec += evaluation.Rprecision(system.get_best_k(topK), gold_stemmed, cut_off=len(gold_stemmed))

            bpref += evaluation.Bpref(system.get_best_k(topK), gold_stemmed)
    
            for i in range(len(P)):
                P[i] += currentP[i]
                R[i] += currentR[i]
                F1[i] += currentF1[i]


    normalized_P = []
    normalized_R = []
    normalized_F1 = []

    for i in range(len(P)):
        normalized_P.append(P[i] / docs)
        normalized_R.append(R[i] / docs)
        normalized_F1.append(F1[i] / docs)


    print ('Evaluation metrics:'.ljust(20, ' '), 'Precision @k'.ljust(20, ' '), 'Recall @k'.ljust(20, ' '), 'F1-score @k')

    for i in range(0, topK):
        print(''.ljust(20, ' '), \
            'Pr@{}'.format(i + 1).ljust(6, ' '), '{0:.3f}'.format(normalized_P[i]).ljust(13, ' '), \
            'Re@{}'.format(i + 1).ljust(6, ' '), '{0:.3f}'.format(normalized_R[i]).ljust(13, ' '), \
            'F1@{}'.format(i + 1).ljust(6, ' '), '{0:.3f}'.format(normalized_F1[i]))


input_data = sys.argv[1]
output_data = sys.argv[2]

topK = 10
window = 10

process(input_data, output_data, topK, window)