from doc_candidates import LoadFile

import networkx as nx

class PositionRank(LoadFile):

    def __init__(self, input_text, window, phrase_type):
        super(PositionRank, self).__init__(input_text=input_text)
        self.graph = nx.Graph()
        self.window = window
        self.phrase_type = phrase_type
        self.valid_pos = ['NN', 'NNS', 'NNP', 'NNPS', 'JJ']

    def build_graph(self, window):
        node_list = []

        # select nodes to be added in the graph
        for word in self.words:
            # if the pos tag corresponding to word is present in the pattern then we will make it a node of the graph..
            if word.pos_pattern in self.valid_pos:
                #  all this properties are inherited from self.words
                node_list.append((word.stemmed_form, word.position, word.sentence_id))

                self.graph.add_node(word.stemmed_form)

        # add edges to graph 
        for i in range(0, len(node_list)):
            for j in range(i+1, len(node_list)):

                position1 = node_list[i][1]
                position2 = node_list[j][1]
                
                node1 = node_list[i][0]
                node2 = node_list[j][0]


                if position1 != position2 and abs(j-i) < window:

                    if self.graph.has_edge(node1, node2):
                        # if edge is already there then just increase the weight of that edge
                        self.graph[node1][node2]['weight'] += 1
                    else:
                        # initialize an edge with edge = 1
                        self.graph.add_edge(node1, node2, weight=1)

    def candidate_selection(self, pos=None, phrase_type='n_grams'):
        if phrase_type=='n_grams':
            self.get_ngrams(n = 4, good_pos = self.valid_pos)
        else:
            # select the longest phrase as candidate keyphrases
            # print("self",self)
            self.get_phrases(self, good_pos = self.valid_pos)

    def candidate_scoring(self, window, update_scoring_method):
        self.build_graph(window)
        self.filter_candidates(max_phrase_length=4, min_word_length=3, valid_punctuation='-.') 

        candidate_list = {}

        for idx, word in enumerate(self.words):
            stem_word = word.stemmed_form
            pos_tag = word.pos_pattern
            position = word.position

            inverse_position = 1.0 / position
            if pos_tag in self.valid_pos:
                if stem_word not in candidate_list:
                    candidate_list[stem_word] = inverse_position
                else:
                    candidate_list[stem_word] += inverse_position
        tot_sum = 0
        for key  in candidate_list:
            tot_sum += candidate_list[key]

        updated_candidate_list = {}
        for key in candidate_list:
            updated_candidate_list[key] = candidate_list[key] / tot_sum

        pagerank_weights = nx.pagerank(self.graph, personalization =  updated_candidate_list, weight='weight')

        for candidate in self.candidates:
            words = candidate.stemmed_form.split()
            total_weight = 0

            for word in words:
                total_weight += pagerank_weights[word]

            self.weights[candidate.stemmed_form] = total_weight

