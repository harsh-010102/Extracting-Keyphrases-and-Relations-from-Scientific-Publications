from __future__ import division

import os
from os.path import isfile, join
import sys
from string import punctuation


from utils import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import re
import codecs
import itertools
from tqdm import tqdm
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter




import nltk
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
nltk.download('averaged_perceptron_tagger')




import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer, BertModel
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch.nn.utils.rnn as rnn_utils
import os 
import pandas as pd
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report

from nltk.stem import PorterStemmer
porter_stemmer = PorterStemmer()
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report


def read_input_file(input_file , mode="r" , encoding='utf-8'):

    text = ""

    if os.path.exists(input_file) :
        with codecs.open(input_file , mode , encoding="utf-8") as f :
            text = f.read()
        
        f.close()
        return text
    
    return None


def read_gold_file(gold_file , mode="r" , encoding='utf-8'):

    text = ""

    if os.path.exists(gold_file) :
        with codecs.open(gold_file , mode , encoding) as f :
            text = f.readlines()
        
        f.close()
        return text
    
    return None



def tokenize(text):
    return text.split()



def get_data(folder_path):
    extracted_data = []

    for file_name in os.listdir(folder_path):
    
        file_path = os.path.join(folder_path, file_name)
    
        with open(file_path, 'r', encoding = 'utf-8') as file:
    
            dataset = file.read()
            text = dataset.split("\n")
    
            for t in text:
                t = t.split(" !!! ")
                keyphrase = t[0].split(" ")
#                 print(keyphrase)
                
                keyphrase = [porter_stemmer.stem(word) for word in keyphrase]
                label = t[1]
                extracted_data.append((keyphrase, label))
            
    return extracted_data


def get_keyphrases(directory):

    count = 0
    keyphrases_list = []

    for filename in os.listdir(directory + "keyphrases"):
        
        keyphrases = []

        with open(os.path.join(directory + "keyphrases", filename), 'r', encoding = "utf-8") as file:
            content = file.readlines()    

        for line in content:
            left_part = line.split("!!!")[0].strip()

            if left_part:
                keyphrases.append(left_part)

        keyphrases_list.append(keyphrases)

    return keyphrases_list
    
def get_labels(directory):

    labels_list = []
    count_H = 0
    count_S = 0

    for filename in os.listdir(directory + "nlis"):

        with open(os.path.join(directory + "nlis", filename), 'r', encoding = "utf-8") as file:
            content = file.readlines()    
        labels = dict()

        for line in content:

            line = line.split("!!!")
            left_part = line[0].strip()
            right_part = line[1].strip()
            label = line[2].strip()
            
            
            if label == "Hyponym":
                count_H += 1

            elif label == "Synonym":
                count_S += 1
       

            keyphrase_tuple = (left_part, right_part)
            labels[keyphrase_tuple] = label

        labels_list.append(labels)

    return labels_list