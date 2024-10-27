import codecs
from nltk import pos_tag, word_tokenize, sent_tokenize
import re
from nltk.stem.porter import PorterStemmer
from string import punctuation
porter_stemmer = PorterStemmer()

class Candidate(object):
    """ The data structure for candidates. """

    def __init__(self, surface_form, pos_pattern, stemmed_form, position, sentence_id):
        """

        :param surface_form: the token form as it appears in the text
        :param pos_pattern: the sequence of pos-tags for the candidate
        :param stemmed_form: the stemmed form of the candidate keyphrase
        :param position: its current position in the document
        :param sentence_id: the number of sentence the candidate appears in.
        """

        self.surface_form = surface_form
        """the candidate form occuring in the text"""

        self.pos_pattern = pos_pattern
        """ the part-of-speech of the candidate. """

        self.stemmed_form = stemmed_form
        """ the stemmed form of the candidate. """

        self.position = position
        """ those positions in the document where the candidate occurs. """

        self.sentence_id = sentence_id
        """ the number of the sentence in the document where the candidate occurs. """

class LoadFile(object):
    #: The LoadFile class that provides base functions
    def __init__(self, input_text) -> None:
        self.input_data = input_text
        self.sentences = sent_tokenize(self.input_data)
        self.tokens = [] #: 2D : List of List of tokens
        self.pos_tags = []
        self.stem = []

        self.words = []
        self.candidates = []
        self.weights = {}
        self.gold_keyphrases = []

        for sentence in self.sentences:
            self.tokens.append(word_tokenize(sentence))

        for tokens_list in self.tokens:
            temp_list_1 = []
            temp_list_2 = []
            for key, tag in pos_tag(tokens_list):
                temp_list_1.append(tag)            
            for tokens in tokens_list:
                temp_list_2.append(porter_stemmer.stem(tokens))
            self.pos_tags.append(temp_list_1)
            self.stem.append(temp_list_2)

    def get_doc_words(self):

        count = 1
        for i in range(len(self.tokens)):
            for j in range(len(self.tokens[i])):
                self.words.append(Candidate(self.tokens[i][j], self.pos_tags[i][j], self.stem[i][j], count, i))
                count+=1

    def get_ngrams(self, n, good_pos):
        """
        compute all the ngrams or ngrams with patters found in the document

        :param n: the maximum length of the ngram (default is 3)
        :param good_pos: goodPOS if any word filter is applied, for eg. keep only nouns and adjectives
        :return:
        """

        jump = 0
        #: jump helps to keep track of a word position; its leap is the sentence length
        for i in range(len(self.tokens)):
            if i != 0:
                jump += len(self.tokens[i-1])

            max_len = min(n, len(self.tokens[i]))

            for j in range(len(self.tokens[i])):
                for k in range(j, len(self.tokens[i])):
                    if self.pos_tags[i][k] in good_pos and k-j < max_len and k < (len(self.tokens[i]) - 1):
                        self.candidates.append(Candidate(' '.join(self.tokens[i][j: k+1]), ' '.join(self.pos_tags[i][j:k+1]), ' '.join(self.stem[i][j:k+1]), j+jump, i))
                    else:
                        break

    def filter_candidates(self, stopwords_file=None, max_phrase_length=4, min_word_length=3, valid_punctuation='-.'):
        """
         discard candidates based on various criteria

        :param stopwords_file: a stop-word file that the user wants to input;
        :param max_phrase_length: filter out phrases longer than max_phrase_length
        :param min_word_length:  filter out phrases that contain words shorter than min_word_length
        :param valid_punctuation: keep tokens that contain any of the valid punctuation
        :return:
        """

        stopwords_list = []
        if stopwords_file is None:
            from nltk.corpus import stopwords
            stopwords_list = set(stopwords.words('english'))
        else:
            with codecs.open(stopwords_file, 'rb', encoding='utf-8') as file:
                file.readlines()
            file.close()
            for line in file:
                stopwords_list.append(line)
        
        indices = []
        for i, c in enumerate(self.candidates):
            tokens = c.surface_form.split()
            pos = c.pos_pattern.split()

            #: discard those candidates that contain stopwords
            if set(tokens).intersection(stopwords_list):
                indices.append(i)
                continue

            #: discard candidates longer than max_phrase_length
            if len(tokens) > max_phrase_length:
                indices.append(i)
                continue

            #: discard candidates that contain words shorter that min_word_length
            if min([len(t) for t in tokens]) < min_word_length:
                indices.append(i)
                continue

            #: discard candidates that end in adjectives (including single word adjectives)
            if pos[-1] == 'JJ':
                indices.append(i)
                continue

            #: to check wheather the tokens are( (, ), {, } , [, ])
            if set(tokens).intersection(set(['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-'])):
                indices.append(i)
                continue

            #: discard candidates that contain other characters except letter, digits, and valid punctuation
            for word in tokens:
                letters_set = set([u for u in word])

                if letters_set.issubset(punctuation):
                    indices.append(i)
                    break

                elif re.match(r'^[a-zA-Z0-9%s]*$' % valid_punctuation, word):
                    continue

                else:
                    indices.append(i)
                    break
                
        dels = 0
        for index in indices:
            #:  after deleting one word whole corpus is shifted by one so we change the ofset
            offset = index - dels
            dels += 1
            del self.candidates[offset]

    def get_best_k(self, k=10):
        """
        return top k predicted keyphrases for the current document
        :param k: top keyphrases to be retuned
        :return: top k keyphrases and their weights
        """

        # sort the candidates in reverse order based on their weights
        sorted_weights = sorted(self.weights, key=self.weights.get, reverse=True)

        # return only the k keyphrases
        return sorted_weights[:(min(k, len(sorted_weights)))]