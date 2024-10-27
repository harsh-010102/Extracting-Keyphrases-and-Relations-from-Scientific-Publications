import os
from os.path import isfile, join
import codecs

def preprocess(dataset_dir, split_name, output_dir):
    print("Input Dir :", dataset_dir)
    print("Split :", split_name)
    print("Output Dir :", output_dir)

    input_dir_docs = dataset_dir + "\\" + split_name + "\\"
    files = [f for f in os.listdir(input_dir_docs) if isfile(join(input_dir_docs, f)) and f[-3:] == "txt"]

    def read_input_file(this_file):
        if os.path.exists(this_file):
            with codecs.open(this_file, "r", encoding='utf-8') as f:
                text = f.read()
            f.close()

            text = text.replace("\t", " ").replace("\n", " ")
        else:
            text = None

        return text

    def read_gold_file(this_gold):
        if os.path.exists(this_gold):
            with codecs.open(this_gold, "r", encoding='utf-8') as f:
                gold_list = f.readlines()
            f.close()
            
            keyphrases = dict()
            keyphrases_list = []
            nli = []

            gold_list = [g.replace("\t", " ").replace("\n", " ") for g in gold_list]
            for g in gold_list:
                if g[0] == 'T':
                    g = g.split(" ")
                    keyphrases[g[0]] = ' '.join(g[4:])
                    keyphrases_list.append(' '.join(g[4:]) + "!!! " + str(g[1]))
            
            for g in gold_list:
                if g[0] == 'R':
                    g = g.split(" ")
                    one = g[2][5:]
                    two = g[3][5:]
                    one = keyphrases[one]
                    two = keyphrases[two]
                    nli.append(one + "!!! " + two + "!!!" + " Hyponym" )

            for g in gold_list:
                if g[0] == '*':
                    g = g.split(" ")
                    one = g[2]
                    two = g[3]
                    one = keyphrases[one]
                    two = keyphrases[two]
                    nli.append(one + "!!! " + two + "!!!" + " Synonym")
                
            for g in gold_list:
                if g[0] != 'T' and g[0] != 'R' and g[0] != '*':
                    print("Unknown Entry Detected!!!! Path :", this_gold)
            return keyphrases_list, nli, this_gold
        return None, None, None

    file_names = []
    paragraphs = []
    keyphrases = []
    nlis = []
    for filename in files:
        gold_filename = filename[:-3]
        text = read_input_file(input_dir_docs + filename)
        ks, n, f = read_gold_file(input_dir_docs + gold_filename +'ann')
        paragraphs.append(text)
        keyphrases.append(ks)
        nlis.append(n)
        file_names.append(f)

    print()
    print(len(paragraphs), len(keyphrases), len(nlis), len(files))

    total_keyphrases = 0
    total_nlis = 0
    for i in range(len(paragraphs)):
        total_keyphrases += len(keyphrases[i])
        total_nlis += len(nlis[i])

    print()
    print("total_keyphrases:", total_keyphrases) 
    print("total_nlis:", total_nlis) 

    # for i in range(len(paragraphs)):
    # # for i in range(10):
    #     print()
    #     print(paragraphs[i])
    #     print(keyphrases[i])
    #     print(nlis[i])
    #     print(files[i])


    uniCodeError = []
    for i in range(len(files)):
        file_1 = output_dir +'/' + split_name + "/paragraphs/" + file_names[i].split("\\")[-1].split(".")[-2] + ".txt"
        file_2 = output_dir +'/' + split_name + "/keyphrases/" + file_names[i].split("\\")[-1].split(".")[-2] + ".txt"
        file_3 = output_dir +'/' + split_name + "/nlis/" + file_names[i].split("\\")[-1].split(".")[-2] + ".txt"

        for lst, file_name in zip([[paragraphs[i]], keyphrases[i], nlis[i]], [file_1, file_2, file_3]):
            directory = os.path.dirname(file_name)
            if not os.path.exists(directory):
                os.makedirs(directory)
            try:
                with open(file_name, 'w', encoding='utf-8') as file:
                    file.write('\n'.join(lst))
            except UnicodeEncodeError:
                uniCodeError.append((file_name, lst))

    print("Total Unicode Errors : ", len(uniCodeError))
    for u in uniCodeError:
        print(u[0].split("/")[-1] , u[0].split("/")[-2])
        print(u[1])
        print()

import sys
if __name__ == "__main__":
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print("Provide Three Arguments \"input_dir split_name output_dir\" ")
    
    preprocess(sys.argv[1], sys.argv[2], sys.argv[3])