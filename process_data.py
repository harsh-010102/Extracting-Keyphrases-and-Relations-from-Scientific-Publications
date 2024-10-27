import re
import codecs
import itertools
import os
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from string import punctuation


valid_word_pattern = r'^[a-zA-Z0-9%s]*$'
blacklist_tokens = ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']



def read_input_file(input_file , mode="r" , encoding='utf-8'):
    
    text = ""

    if os.path.exists(input_file) :
        with codecs.open(input_file , mode , encoding) as f :
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



def tokenize(text, encoding):
    
    tokens_list = []
    precoessed_token = word_tokenize(text.lower().decode(encoding))

    for  token in precoessed_token:
        tokens_list.append(token)
    
    return tokens_list



def filter_candidates(tokens, stopwords_file=None, min_word_length=2, valid_punctuation='-', mode='rb', encoding='utf-8'):

    stopwords_list = []

    if stopwords_file is None :
        stopwords_list = stopwords.words('english')
    
    else :
        with codecs.open(stopwords_file , mode , encoding) as f:
            text = f.readlines()
        f.close()
    
        for stop_word in text:
            stopwords_list.append(stop_word)
    
    
    delete_indices = []

    for index , token in enumerate(tokens):

        len_token = len(token)

        if (token in blacklist_tokens) or  (token in stopwords_list) or (len_token < min_word_length):
            delete_indices.append(index)
        
        else:
            letters_list = []

            for letter in token:
                letters_list.append(letter)
            
            letters_set = set(letters_list)

            if letters_set.issubset(punctuation):
                delete_indices.append(index)
            
            elif re.match(valid_word_pattern % valid_punctuation , token):
                pass

            else:
                delete_indices.append(index)

    delete = 0
    
    len_delete_indices = len(delete_indices)

    for i in range(len_delete_indices):
        delete_index = delete_indices[i]

        offset = delete_index - delete
        del tokens[offset]
        delete = delete + 1

    return tokens


def stemming(text):
    porter_stemmer = PorterStemmer()

    stemmed_text = []

    for word in text:
        stemmed_word = porter_stemmer.stem(word)
        stemmed_text.append(stemmed_word)

    return stemmed_text


def iter_data(path_to_data, encoding , mode = 'rb'):

    index = 1

    for file_name in os.listdir(path_to_data):

        index = index + 1
        file_path = path_to_data + file_name

        with open(file_path , mode) as f :
            file_text = f.read()
            strpped_text = file_text.strip()

            tokens = tokenize(strpped_text , encoding)
            filtted_tokens = filter_candidates(tokens)
            stemmed_tokens = stemming(filtted_tokens)

        f.close()
        yield file_path , strpped_text, stemmed_tokens



class MyCorpus(object):

    def __init__(self, path_to_data, dictionary, length=None, encoding='utf-8'):

        self.corpus_dictionary = dictionary
        self.length = length
        self.encoding = encoding
        self.path_to_data = path_to_data
        self.index_to_filename = {}

    def __iter__(self):

        file_index = 0
        data = iter_data(self.path_to_data, self.encoding)

        for file_index, _ , tokens in itertools.islice( data ,  self.length):

            self.index_to_filename[file_index] = file_index
            index = index + 1
            yield self.corpus_dictionary.doc2bow(tokens)

    def __len__(self):
        if self.length is None:
            
            self.length = 0
            for _ in self:
                self.length += 1
        
        return self.length
