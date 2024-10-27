from __future__ import division
<<<<<<< HEAD

import os
from os.path import isfile, join
import sys
from string import punctuation


from utils import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

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
# nltk.download('averaged_perceptron_tagger')




import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset, RandomSampler, SequentialSampler
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer, BertModel, BertForTokenClassification, BertForSequenceClassification
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from transformers.modeling_outputs import TokenClassifierOutput


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score


from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



#################################################################################################################################################
#####################################################       Train KPE      ######################################################################
#################################################################################################################################################


# this function will read the abtract paragraph and corresponding keyphrases list
def get_dataset_train(abstract_file_path , keyphrases_file_path):
    
    abstract_dataset = []
    keyphrases_dataset = []
    
    # fetching file name from the folder 
    files = [f for f in os.listdir(abstract_file_path) if isfile(join(abstract_file_path, f))]
    i = 0
    
    for file in files:
            
        abstract_file = abstract_file_path + "/" + file 
        keyphrases_file = keyphrases_file_path + "/" + file 
        
        abtract = read_input_file(abstract_file)
        keyphrases = read_gold_file(keyphrases_file)
        
        new_keyphrases = []
        
        for key in keyphrases:
            splitted_key = key.split("!!!")
            new_keyphrases.append(splitted_key[0])

        
        abstract_dataset.append(abtract)
        keyphrases_dataset.append(new_keyphrases)
        
        
    return abstract_dataset , keyphrases_dataset


# Function tokenise the paragraph and do masking in the tokenised text and mark the keyphrase index in the masked the paragraph
def masking_and_labeling_train(abstract_dataset , keyphrases_dataset, tokenizer):
        
    tokenized_texts = []
    formatted_labels_list = []

    for abstract, keyphrase_list in zip(abstract_dataset, keyphrases_dataset):
        tokenized_abstract = tokenizer.tokenize(abstract)
        tokenized_texts.append(tokenized_abstract)

        formatted_label = [0] * len(tokenized_abstract) 

        # marking the keyphrases
        for keyphrase in keyphrase_list:
            keyphrase_tokens = tokenizer.tokenize(keyphrase)
            for i in range(len(tokenized_abstract) - len(keyphrase_tokens) + 1):
                if tokenized_abstract[i:i + len(keyphrase_tokens)] == keyphrase_tokens:
                    for j in range(i, i + len(keyphrase_tokens)):
                        formatted_label[j] = 1  
    #                 break
        formatted_labels_list.append(formatted_label)
    
    return tokenized_texts , formatted_labels_list


def dataset_train(tokenized_texts , formatted_labels_list, tokenizer):
    max_length = max(len(txt) for txt in tokenized_texts)
    input_ids = torch.tensor([tokenizer.convert_tokens_to_ids(txt + ['[PAD]'] * (max_length - len(txt))) for txt in tokenized_texts])

    labels = torch.tensor([lbl + [0] * (max_length - len(lbl)) for lbl in formatted_labels_list])
    
    masks = torch.where(input_ids != 0, torch.tensor(1), torch.tensor(0))
    
    return input_ids , labels, masks


# converting the logit into prediction sequence and finding predicted keyphrase and original keyphrase
def return_keyphrase_train(sentence , logits_list , original_sequence , tokenizer , id_to_token):
    
    # converting the logit  into prediction sequnence
    predited_sequence = []
    batch_size = len(logits_list)
    
    for b in range(batch_size):
        
        batch_res = []
        logits_element = logits_list[b]
        
        for logit in logits_element:
        
            zero_class_logit = logit[0]
            one_class_logit = logit[1]

            if zero_class_logit > one_class_logit:
                batch_res.append(0)
            else:
                batch_res.append(1)
                
        predited_sequence.append(batch_res)
    
    # finding the predicted keyphrases
    predicted_keyphrases = []
    
    for b in range(batch_size):
        
        pred_seq = predited_sequence[b]
        sent = sentence[b]        
        sent_len = len(sent)
        
        pred_phrase = []        
        list_of_pred_phrase = []
        
        for index in range(sent_len):
            
            logit = pred_seq[index]
            word_index = sent[index]
            word = id_to_token[word_index]
            
            if logit == 0:
                
                if len(pred_phrase) < 1:
                    continue
                
                decoded_sentence = tokenizer.convert_tokens_to_string(pred_phrase)       
                list_of_pred_phrase.append(decoded_sentence)
                
                pred_phrase = []
            
            else:
                pred_phrase.append(word)
            
        if len(pred_phrase) > 0:
            decoded_sentence = tokenizer.convert_tokens_to_string(pred_phrase)       
            list_of_pred_phrase.append(decoded_sentence)
            
        predicted_keyphrases.append(list_of_pred_phrase)  
        
        
    # finding the original keyphrases
    original_keyphrases = []
    
    for b in range(batch_size):
        
        original_seq = original_sequence[b]
        sent = sentence[b]        
        sent_len = len(sent)
        
        original_phrase = []        
        list_of_original_phrase = []
        
        for index in range(sent_len):
            
            logit = original_seq[index]
            word_index = sent[index]
            word = id_to_token[word_index]
            
            if logit == 0:
                
                if len(original_phrase) < 1:
                    continue
                
                decoded_sentence = tokenizer.convert_tokens_to_string(original_phrase)       
                list_of_original_phrase.append(decoded_sentence)
                
                original_phrase = []
            
            else:
                original_phrase.append(word)
            
        if len(original_phrase) > 0:
            decoded_sentence = tokenizer.convert_tokens_to_string(original_phrase)       
            list_of_original_phrase.append(decoded_sentence)
            
        original_keyphrases.append(list_of_original_phrase)
    
    return predicted_keyphrases , original_keyphrases
    


def train():

    print("=======================================   Keyphrases Extraction Training ========================================================")
    train_abstract_folder_path = r'preprocessed_dataset\train\paragraphs'
    train_keyphrases_folder_path = r'preprocessed_dataset\train\keyphrases'
    
    val_abstract_folder_path = r'preprocessed_dataset\val\paragraphs'
    val_keyphrases_folder_path = r'preprocessed_dataset\val\keyphrases'
    
    test_abstract_folder_path = r'preprocessed_dataset\test\paragraphs'
    test_keyphrases_folder_path = r'preprocessed_dataset\test\keyphrases'

    train_abstract , train_keyphrases = get_dataset_train(train_abstract_folder_path , train_keyphrases_folder_path)
    val_abstract , val_keyphrases = get_dataset_train(val_abstract_folder_path , val_keyphrases_folder_path)
    test_abstract , test_keyphrases = get_dataset_train(test_abstract_folder_path , test_keyphrases_folder_path)

    # print("The size of Train: " , len(train_abstract))
    # print("The size of Validation: " , len(val_abstract))
    # print("The size of Test: " , len(test_abstract))

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    train_tokenized_texts , train_labels = masking_and_labeling_train(train_abstract , train_keyphrases, tokenizer)
    val_tokenized_texts , val_labels = masking_and_labeling_train(val_abstract , val_keyphrases, tokenizer)
    test_tokenized_texts , test_labels = masking_and_labeling_train(test_abstract , test_keyphrases, tokenizer)

    # print(len(train_tokenized_texts))

    train_inputs , train_labels , train_mask = dataset_train(train_tokenized_texts , train_labels, tokenizer)
    val_inputs , val_labels , val_mask = dataset_train(val_tokenized_texts , val_labels, tokenizer)
    test_inputs  , test_labels , test_mask = dataset_train(test_tokenized_texts , test_labels, tokenizer)


    id_to_token = {token_id: token for token, token_id in tokenizer.vocab.items()}
    # print(len(id_to_token))

    train_data = TensorDataset(train_inputs, train_mask, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=8)


    validation_data = TensorDataset(val_inputs, val_mask, val_labels)
    validation_sampler = RandomSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=8)


    test_data = TensorDataset(test_inputs, test_mask, test_labels)
    test_sampler = RandomSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=8)

    model = BertForTokenClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=2,
        output_attentions=False,
        output_hidden_states=False
    )

    
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=5e-5)


    epochs = 1
    best_f1_score  = 0.0
    best_recall = 0.0
    best_precision = 0.0


    for epoch in range(epochs):
        
        
        model.train()
        total_loss = 0
        
        for batch in tqdm(train_dataloader, desc="Epoch {}/{}".format(epoch + 1 , epochs )):
            
            batch = tuple(t.to(device) for t in batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[2]}
            
            optimizer.zero_grad()
            outputs = model(**inputs)
            
    #         logits_list = outputs.logits.tolist()

            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_dataloader)
        print("Average training loss: {:.4f}".format(avg_train_loss))

        model.eval()
        eval_loss = 0
        nb_eval_steps = 0
        
        predicted_keyphrases_list = []
        original_keyphrases_list = []

        for batch in tqdm(validation_dataloader, desc="Validation"):
            batch = tuple(t.to(device) for t in batch)
            with torch.no_grad():
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[2]}
                outputs = model(**inputs)
                tmp_eval_loss = outputs.loss
                eval_loss += tmp_eval_loss.mean().item()
                
                predicted_labels = torch.argmax(outputs.logits, axis=-1)
                predicted_keyphrases , original_keyphrases = return_keyphrase_train(batch[0].tolist() , outputs.logits.tolist() , batch[2].tolist(), tokenizer , id_to_token)
                
                for keyphrase in predicted_keyphrases:
                    predicted_keyphrases_list.append(keyphrase)
                
                for orig_keyphrase in original_keyphrases:
                    original_keyphrases_list.append(orig_keyphrase)
                
            nb_eval_steps += 1

        avg_val_loss = eval_loss / len(validation_dataloader)
        print("Average validation loss: {:.4f}".format(avg_val_loss))
        
        total_prediction = 0
        total_original = 0
        total_correct_prediction = 0
        total_docs = 0
        
        preicision = 0.0
        recall = 0.0
        f1_score = 0.0
        
        for index in range(len(original_keyphrases_list)):
            
            predicted = predicted_keyphrases_list[index]
            original = original_keyphrases_list[index]
            
            set_predicted = set(predicted)
            set_original = set(original)
            
            correct_prediction = set_original.intersection(set_predicted)
            
            count_prediction = len(set_predicted)
            count_original = len(set_original)
            count_correct_prediction = len(correct_prediction)
            
            docs_precision = count_correct_prediction / count_prediction  if count_prediction > 0 else 0.0
            docs_recall = count_correct_prediction / count_original  if count_original > 0  else 0.0
            docs_f1_score = 2 * docs_precision * docs_recall / (docs_precision + docs_recall) if (docs_precision + docs_recall) > 0.0 else 0.0
            
            preicision += docs_precision
            recall += docs_recall
            f1_score += docs_f1_score

            total_docs += 1
        
        final_precision = preicision/total_docs
        final_recall = recall/total_docs
        final_f1_score = f1_score/total_docs

        
        if final_f1_score > best_f1_score:
            best_precision = final_precision
            best_recall = final_recall
            best_f1_score = final_f1_score

            torch.save(model.state_dict(), 'keyphrases_best_model.pth')
            
        print("Validation Precision : {:.4f}".format(final_precision))
        print("Validation recall : {:.4f}".format(final_recall))    
        print("Validation f1_score : {:.4f}".format(final_f1_score))    
        print("================================================================================")


    print("================================= Validation Result =================================")
    print("Best Validation Precision : {:.4f}".format(best_precision))
    print("Best Validation recall : {:.4f}".format(best_recall))    
    print("Validation f1_score : {:.4f}".format(best_f1_score))  
    print("=====================================================================================")


    best_model = BertForTokenClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=2,  # binary classification (keyphrase or not)
        output_attentions=False,
        output_hidden_states=False
    ).to(device)
    best_model.load_state_dict(torch.load('keyphrases_best_model.pth'))

    best_model.eval()
    test_loss = 0
    nb_test_steps = 0

    predicted_keyphrases_list = []
    original_keyphrases_list = []

    for batch in tqdm(test_dataloader, desc="Test"):
        
        batch = tuple(t.to(device) for t in batch)
        
        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[2]}
            outputs = best_model(**inputs)
            tmp_test_loss = outputs.loss
            test_loss += tmp_test_loss.mean().item()

            predicted_labels = torch.argmax(outputs.logits, axis=-1)
            predicted_keyphrases , original_keyphrases = return_keyphrase_train(batch[0].tolist() , outputs.logits.tolist() , batch[2].tolist(),  tokenizer , id_to_token)

            for keyphrase in predicted_keyphrases:
                predicted_keyphrases_list.append(keyphrase)

            for orig_keyphrase in original_keyphrases:
                original_keyphrases_list.append(orig_keyphrase)

        nb_test_steps += 1

    avg_test_loss = test_loss / len(test_dataloader)


    preicision = 0.0
    recall = 0.0
    f1_score = 0.0
    total_docs = 0

    for index in range(len(original_keyphrases_list)):

        predicted = predicted_keyphrases_list[index]
        original = original_keyphrases_list[index]

        set_predicted = set(predicted)
        set_original = set(original)

        correct_prediction = set_original.intersection(set_predicted)

        count_prediction = len(set_predicted)
        count_original = len(set_original)
        count_correct_prediction = len(correct_prediction)

        docs_precision = count_correct_prediction / count_prediction  if count_prediction > 0 else 0.0
        docs_recall = count_correct_prediction / count_original  if count_original > 0  else 0.0
        docs_f1_score = 2 * docs_precision * docs_recall / (docs_precision + docs_recall) if (docs_precision + docs_recall) > 0.0 else 0.0

        preicision += docs_precision
        recall += docs_recall
        f1_score += docs_f1_score

        total_docs += 1

    final_precision = preicision/total_docs
    final_recall = recall/total_docs
    final_f1_score = f1_score/total_docs

    print("==========================   Test  Results     ==========================================")
    print("Test Precision : {:.4f}".format(final_precision))
    print("Test recall : {:.4f}".format(final_recall))    
    print("Test f1_score : {:.4f}".format(final_f1_score))  
    print("============================================================================================\n\n")


    # NER 
    print("=======================================   Keyphrases NER Training ========================================================")

    train_data = get_data(r'preprocessed_dataset\train\keyphrases')
    val_data = get_data(r'preprocessed_dataset\val\keyphrases')
    test_data = get_data(r'preprocessed_dataset\test\keyphrases')


    word2idx = dict()
    word2idx['pad'] = 0
    word2idx['unk'] = 1
    for text, label in train_data:
        for word in text:
            if word not in word2idx:
                word2idx[word] = len(word2idx)

    class CustomDataset(Dataset):
        
        def __init__(self, data, word2idx):
            self.data = data 
            self.word2idx = word2idx
            self.num_classes = 3 

        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            words, label = self.data[idx]
            idx_list = [self.word2idx.get(word, 1) for word in words]  # Use 1 for 'unk' if word not in word2idx
            label_idx = 0 if label == 'Process' else 1 if label == 'Task' else 2
            return torch.tensor(idx_list), torch.tensor(label_idx)
   
    def collate_fn(batch):
        batch.sort(key=lambda x: len(x[0]), reverse=True)
        sequences, labels = zip(*batch)
        padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)
        return padded_sequences, torch.tensor(labels)
    

    train_dataset = CustomDataset(train_data, word2idx)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_dataset = CustomDataset(val_data, word2idx)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

    test_dataset = CustomDataset(test_data, word2idx)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    class BiLSTMClassifier(nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            super(BiLSTMClassifier, self).__init__()
            self.embedding = nn.Embedding(input_size, hidden_size)
            self.bilstm = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=True)
            self.fc = nn.Linear(hidden_size * 2, num_classes)
            
        def forward(self, x):
            x = self.embedding(x)
            x, _ = self.bilstm(x)
            x = self.fc(x[:, -1, :])
            return x

    # Define hyperparameters
    input_size = len(word2idx)
    hidden_size = 128
    num_classes = 3
    learning_rate = 0.001
    num_epochs = 10

    # Initialize the model, loss function, and optimizer
    model = BiLSTMClassifier(input_size, hidden_size, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()  
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    val_losses, train_losses = [], []
    val_accuracy = []

    min_loss = float('inf')
    best_model_param = None

    for epoch in range(num_epochs):
        t_loss = 0.0
        v_loss = 0.0
        correct_pred = 0
        tot_pred = 0
        model.train()

        print("Epoch", epoch + 1)

        for input_batch, output_batch in tqdm(train_dataloader, total=len(train_dataloader), desc="Training"):
            input_batch = input_batch.to(device)
            output_batch = output_batch.to(device)

            optimizer.zero_grad()

            output = model(input_batch)

            loss = criterion(output, output_batch)

            loss.backward()
            optimizer.step()

            t_loss += loss.item()

        t_loss = t_loss / len(train_dataloader)
        train_losses.append(t_loss)


        with torch.no_grad():
            model.eval()

            for input_batch, output_batch in tqdm(val_dataloader, total=len(val_dataloader), desc="Validation"):
                input_batch = input_batch.to(device)
                output_batch = output_batch.to(device)

                output = model(input_batch)

                loss = criterion(output, output_batch)

                v_loss += loss.item()

                _, predicted = torch.max(output, 1)

                correct_pred += (predicted == output_batch).sum().item()
                tot_pred += output_batch.size(0)

            v_loss = v_loss / len(val_dataloader)
            val_losses.append(v_loss)

            if v_loss < min_loss:
                best_model_param = model.state_dict()
                min_loss = v_loss

            val_accuracy.append(correct_pred / tot_pred)
            print("train loss: ", t_loss)
            print("Val Acc:  ", correct_pred / tot_pred)
            print("val loss:  ", v_loss)

    
    # find the classification accuracy on test set
    model.load_state_dict(best_model_param)
    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        model.eval()

        correct_pred = 0
        tot_pred = 0

        for input_batch, output_batch in tqdm(test_dataloader, total = len(test_dataloader), desc = "Testing"):
            output = model(input_batch.to(device))
            
            output_batch = output_batch.to(device).to(torch.float32)

            _, predicted = torch.max(output, 1)

            correct_pred += (predicted == output_batch).sum().item()
            tot_pred += output_batch.size(0)
            
            true_labels.extend(output_batch.cpu().numpy())
            predicted_labels.extend(predicted.cpu().numpy())
            
            accuracy = correct_pred / tot_pred

            classification_rep = classification_report(true_labels, predicted_labels, zero_division=1)
    
    print("\n============================  Test Results ====================================")
    print("Accuracy is ", accuracy)
    print("\n")
    print(classification_rep)
    print("==================================================================================\n")


    model.load_state_dict(best_model_param)
    torch.save(model.state_dict(), "ner.pt")



    # NLI
    print("=======================================   Keyphrases NLI  Training ========================================================")

    preprocess_dataset_train = r'preprocessed_dataset\train\\'
    preprocess_dataset_eval = r'preprocessed_dataset\val\\'
    preprocess_dataset_test = r'preprocessed_dataset\test\\'
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    model_nli = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

    model_nli = model_nli.to(device)

    keyphrases_train = get_keyphrases(preprocess_dataset_train)
    keyphrases_eval = get_keyphrases(preprocess_dataset_eval)
    keyphrases_test = get_keyphrases(preprocess_dataset_test)


    labels_train = get_labels(preprocess_dataset_train)
    labels_eval = get_labels(preprocess_dataset_eval)
    labels_test = get_labels(preprocess_dataset_test)

    class NLI_Dataset(Dataset):
        
        def __init__(self, keyphrases, labels, train):
            self.keyphrases = keyphrases
            self.labels = labels
            self.train = train
            
            self.keyphrases_dataset = []
            self.nli_labels_list = []

            self.generate_pairs()


        def __len__(self):
            return len(self.keyphrases_dataset)


        def __getitem__(self, idx):
            return self.keyphrases_dataset[idx], self.nli_labels_list[idx]
        

        def pad_sequence_priv(self, sequence, max_length):
            padding_length = max_length - sequence.shape[1]
            if padding_length > 0:
                return torch.nn.functional.pad(sequence, (0, padding_length))
            else:
                return sequence


        def generate_pairs(self):
            max_seq_length = 15
            
            count_s = 0
            count_h = 0
            count_n = 0
            
            for k in range(len(self.keyphrases)):
                
                keyphrases_list = self.keyphrases[k]
                nli_labels = []
                count_none = 2
                
            
                for i in range(len(keyphrases_list)):
                    for j in range(len(keyphrases_list)):
                        
                        if i == j:
                            continue
                        
                        tokenized_keyphrase_pair = tokenizer(self.keyphrases[k][i], self.keyphrases[k][j], padding=True, truncation=True, return_tensors="pt")

                        padded_input_ids = self.pad_sequence_priv(tokenized_keyphrase_pair['input_ids'], max_seq_length)
                        padded_attention_mask = self.pad_sequence_priv(tokenized_keyphrase_pair['attention_mask'], max_seq_length)
                        padded_token_type_ids = self.pad_sequence_priv(tokenized_keyphrase_pair['token_type_ids'], max_seq_length)

                        padded_tokenized_keyphrase_pair = {
                            'input_ids': padded_input_ids,
                            'attention_mask': padded_attention_mask,
                            'token_type_ids': padded_token_type_ids
                        }

                        keyphrase_pair = (self.keyphrases[k][i], self.keyphrases[k][j])
        
                        if keyphrase_pair in self.labels[k]:
                            
                            if self.labels[k][keyphrase_pair] == 'Synonym':
                                if j < i:
                                    continue
                                
                                self.nli_labels_list.append(1)
                                count_s+=1
                                self.keyphrases_dataset.append(padded_tokenized_keyphrase_pair)

                            elif self.labels[k][keyphrase_pair] == 'Hyponym':
                                self.nli_labels_list.append(2)
                                count_h+=1
                                self.keyphrases_dataset.append(padded_tokenized_keyphrase_pair)
                        
                        elif self.train and count_none > 0:
                            count_none -= 1
                            count_n+= 1
                            
                            self.nli_labels_list.append(0)
                            self.keyphrases_dataset.append(padded_tokenized_keyphrase_pair)
                        
                        elif not self.train and count_none > 0:
                            count_none -= 1
                            count_n += 1
                            self.nli_labels_list.append(0)
                            self.keyphrases_dataset.append(padded_tokenized_keyphrase_pair)
            
            print("count_s :", count_s , "count_h :", count_h , "count_n :", count_n)


    train_dataset = NLI_Dataset(keyphrases_train, labels_train, train = True)
    eval_dataset = NLI_Dataset(keyphrases_eval, labels_eval, train = True)
    test_dataset = NLI_Dataset(keyphrases_test, labels_test, train = False)


    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


        
    optimizer = torch.optim.AdamW(model_nli.parameters(), lr=2e-5)
    loss_fn = torch.nn.CrossEntropyLoss()

    epochs = 1
    
    for epoch in range(epochs):
    
        train_loss = 0
        eval_loss = 0
        best_eval_loss = float('inf')
        model_nli.train()
    
        for token_input, labels in tqdm(train_loader):
    
            labels = labels.to(device)
            input_ids = token_input['input_ids'].squeeze(1).to(device)
            attention_mask = token_input['attention_mask'].squeeze(1).to(device)
            token_type_ids = token_input['token_type_ids'].squeeze(1).to(device)

            optimizer.zero_grad()
            outputs = model_nli(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels)
    
            loss = outputs.loss
            loss.backward()
    
            optimizer.step()
    
            train_loss += loss.item()

        model_nli.eval()
        correct = 0
        total = 0
    
        with torch.no_grad():
            for token_input, labels in tqdm(eval_loader):
           
                labels = labels.to(device)
                input_ids = token_input['input_ids'].squeeze(1).to(device)
                attention_mask = token_input['attention_mask'].squeeze(1).to(device)
                token_type_ids = token_input['token_type_ids'].squeeze(1).to(device)

                outputs = model_nli(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels)
                eval_loss += outputs.loss.item()

                _, predicted = torch.max(outputs.logits, 1)
           
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        train_loss /= len(train_loader)
        eval_loss /= len(eval_loader)
        
        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            best_model_state = model_nli.state_dict().copy()


        print(f"Epoch {epoch+1}/{epochs}: Train Loss {train_loss:.4f} | Eval Loss {eval_loss:.4f}")
    
    torch.save(best_model_state, "nli.pt")

    model_nli.load_state_dict(best_model_state)

    model_nli.eval()
    predictions_nli = []
    
    true_labels_nli = []
    
    with torch.no_grad():
        for token_input, labels in tqdm(test_loader):
            
            token_input = token_input
            labels = labels.to(device)

            input_ids = token_input['input_ids'].squeeze(1).to(device)
            attention_mask = token_input['attention_mask'].squeeze(1).to(device)  
            token_type_ids = token_input['token_type_ids'].squeeze(1).to(device) 

            outputs = model_nli(input_ids, attention_mask=attention_mask)
            _, predicted = torch.max(outputs.logits, 1)

            predictions_nli.extend(predicted.tolist())
            true_labels_nli.extend(labels.tolist())


    print("\n\n=================================  Test Result================================================")
    print(classification_report(true_labels_nli, predictions_nli))
    print("===================================================================================================")





#################################################################################################################################################
#####################################################       Eval        #########################################################################
#################################################################################################################################################

def get_dataset_eval(abstract_file_path , keyphrases_file_path):
    
    abstract_dataset = []
    keyphrases_dataset = []
    
    files = [f for f in os.listdir(abstract_file_path) if isfile(join(abstract_file_path, f))]
    i =0
    
    
    for file in files:
            
        abstract_file = abstract_file_path + "/" + file 
        keyphrases_file = keyphrases_file_path + "/" + file 
        
        abtract = read_input_file(abstract_file)
        keyphrases = read_gold_file(keyphrases_file)
        
        new_keyphrases = []
        
        for key in keyphrases:
            splitted_key = key.split("!!!")
            new_keyphrases.append(splitted_key[0])
        
#         if  i ==0:
#             i = 2
#             print(file)
#             print(type(keyphrases))
#             print(keyphrases)
#             print("==================================")
#             print(new_keyphrases)
#             print(type(new_keyphrases))
        
        abstract_dataset.append(abtract)
        keyphrases_dataset.append(new_keyphrases)
        
        
    return abstract_dataset , keyphrases_dataset

def masking_and_labeling_eval(abstract_dataset , keyphrases_dataset, tokenizer):
        
    tokenized_texts = []
    formatted_labels_list = []

    for abstract, keyphrase_list in zip(abstract_dataset, keyphrases_dataset):
        tokenized_abstract = tokenizer.tokenize(abstract)
        tokenized_texts.append(tokenized_abstract)

        formatted_label = [0] * len(tokenized_abstract) 

        # marking the keyphrases
        for keyphrase in keyphrase_list:
            keyphrase_tokens = tokenizer.tokenize(keyphrase)
            for i in range(len(tokenized_abstract) - len(keyphrase_tokens) + 1):
                if tokenized_abstract[i:i + len(keyphrase_tokens)] == keyphrase_tokens:
                    for j in range(i, i + len(keyphrase_tokens)):
                        formatted_label[j] = 1  
    #                 break
        formatted_labels_list.append(formatted_label)
    
    return tokenized_texts , formatted_labels_list


def dataset_eval(tokenized_texts , formatted_labels_list, tokenizer):
    max_length = max(len(txt) for txt in tokenized_texts)
    input_ids = torch.tensor([tokenizer.convert_tokens_to_ids(txt + ['[PAD]'] * (max_length - len(txt))) for txt in tokenized_texts])

    labels = torch.tensor([lbl + [0] * (max_length - len(lbl)) for lbl in formatted_labels_list])
    
    masks = torch.where(input_ids != 0, torch.tensor(1), torch.tensor(0))
    
    return input_ids , labels, masks


def return_keyphrase_eval(sentence , logits_list , original_sequence , tokenizer , id_to_token):
    
    # converting the logit  into prediction sequnence
    predited_sequence = []
    batch_size = len(logits_list)
    
    for b in range(batch_size):
        
        batch_res = []
        logits_element = logits_list[b]
        
        for logit in logits_element:
        
            zero_class_logit = logit[0]
            one_class_logit = logit[1]

            if zero_class_logit > one_class_logit:
                batch_res.append(0)
            else:
                batch_res.append(1)
                
        predited_sequence.append(batch_res)
    
    # finding the predicted keyphrases
    predicted_keyphrases = []
    
    for b in range(batch_size):
        
        pred_seq = predited_sequence[b]
        sent = sentence[b]        
        sent_len = len(sent)
        
        pred_phrase = []        
        list_of_pred_phrase = []
        
        for index in range(sent_len):
            
            logit = pred_seq[index]
            word_index = sent[index]
            word = id_to_token[word_index]
            
            if logit == 0:
                
                if len(pred_phrase) < 1:
                    continue
                
                decoded_sentence = tokenizer.convert_tokens_to_string(pred_phrase)       
                list_of_pred_phrase.append(decoded_sentence)
                
                pred_phrase = []
            
            else:
                pred_phrase.append(word)
            
        if len(pred_phrase) > 0:
            decoded_sentence = tokenizer.convert_tokens_to_string(pred_phrase)       
            list_of_pred_phrase.append(decoded_sentence)
            
        predicted_keyphrases.append(list_of_pred_phrase)  
        
        
    # finding the original keyphrases
    original_keyphrases = []
    
    for b in range(batch_size):
        
        original_seq = original_sequence[b]
        sent = sentence[b]        
        sent_len = len(sent)
        
        original_phrase = []        
        list_of_original_phrase = []
        
        for index in range(sent_len):
            
            logit = original_seq[index]
            word_index = sent[index]
            word = id_to_token[word_index]
            
            if logit == 0:
                
                if len(original_phrase) < 1:
                    continue
                
                decoded_sentence = tokenizer.convert_tokens_to_string(original_phrase)       
                list_of_original_phrase.append(decoded_sentence)
                
                original_phrase = []
            
            else:
                original_phrase.append(word)
            
        if len(original_phrase) > 0:
            decoded_sentence = tokenizer.convert_tokens_to_string(original_phrase)       
            list_of_original_phrase.append(decoded_sentence)
            
        original_keyphrases.append(list_of_original_phrase)
    
    return predicted_keyphrases , original_keyphrases
    


def eval(kpe_model_path, ner_model_path, nli_model_path):
    print("=======================================   Keyphrases Extraction Evalution ========================================================")

    val_abstract_folder_path = r'preprocessed_dataset\val\paragraphs'
    val_keyphrases_folder_path = r'preprocessed_dataset\val\keyphrases'
    
    test_abstract_folder_path = r'preprocessed_dataset\test\paragraphs'
    test_keyphrases_folder_path = r'preprocessed_dataset\test\keyphrases'

    val_abstract , val_keyphrases = get_dataset_eval(val_abstract_folder_path , val_keyphrases_folder_path)
    test_abstract , test_keyphrases = get_dataset_eval(test_abstract_folder_path , test_keyphrases_folder_path)

    # print("The size of Validation: " , len(val_abstract))
    # print("The size of Test: " , len(test_abstract))

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    
    val_tokenized_texts , val_labels = masking_and_labeling_eval(val_abstract , val_keyphrases, tokenizer)
    test_tokenized_texts , test_labels = masking_and_labeling_eval(test_abstract , test_keyphrases, tokenizer)

    id_to_token = {token_id: token for token, token_id in tokenizer.vocab.items()}
    # print(len(id_to_token))


    val_inputs , val_labels , val_mask = dataset_eval(val_tokenized_texts , val_labels, tokenizer)
    test_inputs  , test_labels , test_mask = dataset_eval(test_tokenized_texts , test_labels, tokenizer)


    validation_data = TensorDataset(val_inputs, val_mask, val_labels)
    validation_sampler = RandomSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=8)


    test_data = TensorDataset(test_inputs, test_mask, test_labels)
    test_sampler = RandomSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=8)


    best_model = BertForTokenClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=2, 
        output_attentions=False,
        output_hidden_states=False
    ).to(device)

    # optimizer = AdamW(model.parameters(), lr=5e-5)

    best_model.load_state_dict(torch.load(kpe_model_path))
    # print(model_path)

    best_model.eval()

    val_loss = 0
    nb_val_steps = 0

    predicted_keyphrases_list = []
    original_keyphrases_list = []

    for batch in tqdm(validation_dataloader, desc="Validation"):
        
        batch = tuple(t.to(device) for t in batch)
        
        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[2]}
            outputs = best_model(**inputs)
            tmp_val_loss = outputs.loss
            val_loss += tmp_val_loss.mean().item()

            predicted_labels = torch.argmax(outputs.logits, axis=-1)
            predicted_keyphrases , original_keyphrases = return_keyphrase_eval(batch[0].tolist() , outputs.logits.tolist() , batch[2].tolist(),  tokenizer , id_to_token)

            for keyphrase in predicted_keyphrases:
                predicted_keyphrases_list.append(keyphrase)

            for orig_keyphrase in original_keyphrases:
                original_keyphrases_list.append(orig_keyphrase)

        nb_val_steps += 1

    avg_val_loss = val_loss / len(test_dataloader)

    preicision = 0.0
    recall = 0.0
    f1_score = 0.0
    total_docs = 0

    for index in range(len(original_keyphrases_list)):

        predicted = predicted_keyphrases_list[index]
        original = original_keyphrases_list[index]

        set_predicted = set(predicted)
        set_original = set(original)

        correct_prediction = set_original.intersection(set_predicted)

        count_prediction = len(set_predicted)
        count_original = len(set_original)
        count_correct_prediction = len(correct_prediction)

        docs_precision = count_correct_prediction / count_prediction  if count_prediction > 0 else 0.0
        docs_recall = count_correct_prediction / count_original  if count_original > 0  else 0.0
        docs_f1_score = 2 * docs_precision * docs_recall / (docs_precision + docs_recall) if (docs_precision + docs_recall) > 0.0 else 0.0

        preicision += docs_precision
        recall += docs_recall
        f1_score += docs_f1_score

        total_docs += 1

    final_precision = preicision/total_docs
    final_recall = recall/total_docs
    final_f1_score = f1_score/total_docs

    print("\n\n=======================================   Validation   Results     ================================================")
    print("Validation Precision : {:.4f}".format(final_precision))
    print("Validation recall : {:.4f}".format(final_recall))    
    print("Validation f1_score : {:.4f}".format(final_f1_score))  
    print("=======================================================================================================================\n\n")

    best_model.eval()
    test_loss = 0
    nb_test_steps = 0

    predicted_keyphrases_list = []
    original_keyphrases_list = []

    for batch in tqdm(test_dataloader, desc="Test"):
        
        batch = tuple(t.to(device) for t in batch)
        
        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[2]}
            outputs = best_model(**inputs)
            tmp_test_loss = outputs.loss
            test_loss += tmp_test_loss.mean().item()

            predicted_labels = torch.argmax(outputs.logits, axis=-1)
            predicted_keyphrases , original_keyphrases = return_keyphrase_eval(batch[0].tolist() , outputs.logits.tolist() , batch[2].tolist(),  tokenizer , id_to_token)

            for keyphrase in predicted_keyphrases:
                predicted_keyphrases_list.append(keyphrase)

            for orig_keyphrase in original_keyphrases:
                original_keyphrases_list.append(orig_keyphrase)

        nb_test_steps += 1

    avg_test_loss = test_loss / len(test_dataloader)
    print("Average Test loss: {:.4f}".format(avg_val_loss))

    total_prediction = 0
    total_original = 0
    total_correct_prediction = 0
    total_docs = 0

    preicision = 0.0
    recall = 0.0
    f1_score = 0.0

    for index in range(len(original_keyphrases_list)):

        predicted = predicted_keyphrases_list[index]
        original = original_keyphrases_list[index]

        set_predicted = set(predicted)
        set_original = set(original)

        correct_prediction = set_original.intersection(set_predicted)

        count_prediction = len(set_predicted)
        count_original = len(set_original)
        count_correct_prediction = len(correct_prediction)

        docs_precision = count_correct_prediction / count_prediction  if count_prediction > 0 else 0.0
        docs_recall = count_correct_prediction / count_original  if count_original > 0  else 0.0
        docs_f1_score = 2 * docs_precision * docs_recall / (docs_precision + docs_recall) if (docs_precision + docs_recall) > 0.0 else 0.0

        preicision += docs_precision
        recall += docs_recall
        f1_score += docs_f1_score

        total_docs += 1

    final_precision = preicision/total_docs
    final_recall = recall/total_docs
    final_f1_score = f1_score/total_docs


    print("\n\n=====================================   Test  Results     ==================================================")
    print("Validation Precision : {:.4f}".format(final_precision))
    print("Validation recall : {:.4f}".format(final_recall))    
    print("Validation f1_score : {:.4f}".format(final_f1_score))  
    print("==================================================================================================================\n\n")


    print("==========================================   Keyphrases NER Evalution ===========================================================")

    
    train_data = get_data(r'preprocessed_dataset\train\keyphrases')
    val_data = get_data(r'preprocessed_dataset\val\keyphrases')
    test_data = get_data(r'preprocessed_dataset\test\keyphrases')


    word2idx = dict()
    word2idx['pad'] = 0
    word2idx['unk'] = 1
    for text, label in train_data:
        for word in text:
            if word not in word2idx:
                word2idx[word] = len(word2idx)

    class CustomDataset(Dataset):
        def __init__(self, data, word2idx):
            self.data = data 
            self.word2idx = word2idx
            self.num_classes = 3 

        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            words, label = self.data[idx]
            idx_list = [self.word2idx.get(word, 1) for word in words]  # Use 1 for 'unk' if word not in word2idx
            label_idx = 0 if label == 'Process' else 1 if label == 'Task' else 2
            return torch.tensor(idx_list), torch.tensor(label_idx)
        
    def collate_fn(batch):
        batch.sort(key=lambda x: len(x[0]), reverse=True)
        sequences, labels = zip(*batch)
        padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)
        return padded_sequences, torch.tensor(labels)
    

    val_dataset = CustomDataset(val_data, word2idx)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

    test_dataset = CustomDataset(test_data, word2idx)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    class BiLSTMClassifier(nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            super(BiLSTMClassifier, self).__init__()
            self.embedding = nn.Embedding(input_size, hidden_size)
            self.bilstm = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=True)
            self.fc = nn.Linear(hidden_size * 2, num_classes)
            
        def forward(self, x):
            x = self.embedding(x)
            x, _ = self.bilstm(x)
            x = self.fc(x[:, -1, :])
            return x

    # Define hyperparameters
    input_size = len(word2idx)
    hidden_size = 128
    num_classes = 3


    best_ner_model = BiLSTMClassifier(input_size, hidden_size, num_classes).to(device)
    best_ner_model.load_state_dict(torch.load(ner_model_path))

    criterion = nn.CrossEntropyLoss()  

    val_losses =  []
    val_accuracy = []

    correct_pred = 0
    tot_pred = 0 

    with torch.no_grad():
        best_ner_model.eval()
        v_loss = 0.0

        for input_batch, output_batch in tqdm(val_dataloader, total=len(val_dataloader), desc="Validation"):
            input_batch = input_batch.to(device)
            output_batch = output_batch.to(device)

            output = best_ner_model(input_batch)

            loss = criterion(output, output_batch)

            v_loss += loss.item()

            _, predicted = torch.max(output, 1)

            correct_pred += (predicted == output_batch).sum().item()
            tot_pred += output_batch.size(0)

        v_loss = v_loss / len(val_dataloader)
        val_losses.append(v_loss)


        val_accuracy.append(correct_pred / tot_pred)

        print("======================================= Validation Result ===========================================\n")
        print("NER Validation Acc", correct_pred / tot_pred)
        print("NER Validation loss", v_loss)
        print("=======================================================================================================")
    print("\n\n")


    true_labels = []
    predicted_labels = []
    correct_pred = 0
    tot_pred = 0

    with torch.no_grad():
        best_ner_model.eval()

        correct_pred = 0
        tot_pred = 0
        for input_batch, output_batch in tqdm(test_dataloader, total = len(test_dataloader), desc = "Testing"):
            output = best_ner_model(input_batch.to(device))
            # print(type(output), output.shape)
            output_batch = output_batch.to(device).to(torch.float32)

            _, predicted = torch.max(output, 1)

            correct_pred += (predicted == output_batch).sum().item()
            tot_pred += output_batch.size(0)
            
            true_labels.extend(output_batch.cpu().numpy())
            predicted_labels.extend(predicted.cpu().numpy())
            
            accuracy = correct_pred / tot_pred

            classification_rep = classification_report(true_labels, predicted_labels, zero_division=1)
    
    print("======================================== TEST Result =======================================\n")
    print("Accuracy is ", accuracy)
    print("\n\n")
    print(classification_rep)
    print("=========================================================================================\n\n")

    print("\n==========================================   Keyphrases NLI Evalution ===========================================================")

    
    
    model_nli = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
    model_nli = model_nli.to(device)
    
    model_nli.load_state_dict(torch.load(nli_model_path))

    preprocess_dataset_eval = r'preprocessed_dataset\val\\'
    preprocess_dataset_test = r'preprocessed_dataset\test\\'

    keyphrases_eval = get_keyphrases(preprocess_dataset_eval)
    keyphrases_test = get_keyphrases(preprocess_dataset_test)

    labels_eval = get_labels(preprocess_dataset_eval)
    labels_test = get_labels(preprocess_dataset_test)

    class NLIDatasetEval(Dataset):
        
        def __init__(self, keyphrases, labels, train):
            self.keyphrases = keyphrases
            self.labels = labels
            self.train = train
            
            self.keyphrases_dataset = []
            self.nli_labels_list = []

            self.generate_pairs()


        def __len__(self):
            return len(self.keyphrases_dataset)


        def __getitem__(self, idx):
            return self.keyphrases_dataset[idx], self.nli_labels_list[idx]
        

        def pad_sequence_priv(self, sequence, max_length):
            padding_length = max_length - sequence.shape[1]
            if padding_length > 0:
                return torch.nn.functional.pad(sequence, (0, padding_length))
            else:
                return sequence


        def generate_pairs(self):
            max_seq_length = 15
            
            count_s = 0
            count_h = 0
            count_n = 0
            
            for k in range(len(self.keyphrases)):
                
                keyphrases_list = self.keyphrases[k]
                nli_labels = []
                count_none = 2
                

                
                for i in range(len(keyphrases_list)):
                    for j in range(len(keyphrases_list)):
                        
                        if i == j:
                            continue
                        
                        tokenized_keyphrase_pair = tokenizer(self.keyphrases[k][i], self.keyphrases[k][j], padding=True, truncation=True, return_tensors="pt")

                        padded_input_ids = self.pad_sequence_priv(tokenized_keyphrase_pair['input_ids'], max_seq_length)
                        padded_attention_mask = self.pad_sequence_priv(tokenized_keyphrase_pair['attention_mask'], max_seq_length)
                        padded_token_type_ids = self.pad_sequence_priv(tokenized_keyphrase_pair['token_type_ids'], max_seq_length)

                        padded_tokenized_keyphrase_pair = {
                            'input_ids': padded_input_ids,
                            'attention_mask': padded_attention_mask,
                            'token_type_ids': padded_token_type_ids
                        }

                        keyphrase_pair = (self.keyphrases[k][i], self.keyphrases[k][j])
        
                        if keyphrase_pair in self.labels[k]:
                            if self.labels[k][keyphrase_pair] == 'Synonym':
                                
                                if j < i:
                                    continue
                                
                                self.nli_labels_list.append(1)
                                count_s+=1
                                self.keyphrases_dataset.append(padded_tokenized_keyphrase_pair)

                            elif self.labels[k][keyphrase_pair] == 'Hyponym':
                                self.nli_labels_list.append(2)
                                count_h+=1
                                self.keyphrases_dataset.append(padded_tokenized_keyphrase_pair)
                        
                        elif self.train and count_none > 0:
                            count_none -= 1
                            count_n+= 1
                            self.nli_labels_list.append(0)
                            self.keyphrases_dataset.append(padded_tokenized_keyphrase_pair)
                        
                        elif not self.train and count_none > 0:
                            count_none -= 1
                            count_n += 1
                            self.nli_labels_list.append(0)
                            self.keyphrases_dataset.append(padded_tokenized_keyphrase_pair)
                        
    eval_dataset = NLIDatasetEval(keyphrases_eval, labels_eval, train = True)
    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=True)

    test_dataset = NLIDatasetEval(keyphrases_test, labels_test, train = False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model_nli.eval()

    correct = 0
    eval_loss = 0
    total = 0

    with torch.no_grad():
        for token_input, labels in tqdm(eval_loader):

            labels = labels.to(device)
            input_ids = token_input['input_ids'].squeeze(1).to(device)
            attention_mask = token_input['attention_mask'].squeeze(1).to(device)
            token_type_ids = token_input['token_type_ids'].squeeze(1).to(device)

            outputs = model_nli(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels)
            eval_loss += outputs.loss.item()

            _, predicted = torch.max(outputs.logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()


    eval_loss /= len(eval_loader)
    
    print("eval_loss:", eval_loss)
    print("eval_accuracy", correct / total)

    model_nli.eval()
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for token_input, labels in tqdm(test_loader):
            
            token_input = token_input
            labels = labels.to(device)

            input_ids = token_input['input_ids'].squeeze(1).to(device)
            attention_mask = token_input['attention_mask'].squeeze(1).to(device)  
            token_type_ids = token_input['token_type_ids'].squeeze(1).to(device) 

            outputs = model_nli(input_ids, attention_mask=attention_mask)
            _, predicted = torch.max(outputs.logits, 1)

            predictions.extend(predicted.tolist())
            true_labels.extend(labels.tolist())

    print("\n============================ Test result =============================================\n")
    print(classification_report(true_labels, predictions))

    print("\n========================================================================================\n")

#################################################################################################################################################
#####################################################       Test        #########################################################################
#################################################################################################################################################

def get_dataset_test(abstract_file_path , keyphrases_file_path):
    
    abstract_dataset = []
    keyphrases_dataset = []
    
    files = [f for f in os.listdir(abstract_file_path) if isfile(join(abstract_file_path, f))]
    i =0
    
    
    for file in files:
            
        abstract_file = abstract_file_path + "/" + file 
        keyphrases_file = keyphrases_file_path + "/" + file 
        
        abtract = read_input_file(abstract_file)
        keyphrases = read_gold_file(keyphrases_file)
        
        new_keyphrases = []
        
        for key in keyphrases:
            splitted_key = key.split("!!!")
            new_keyphrases.append(splitted_key[0])
        
#         if  i ==0:
#             i = 2
#             print(file)
#             print(type(keyphrases))
#             print(keyphrases)
#             print("==================================")
#             print(new_keyphrases)
#             print(type(new_keyphrases))
        
        abstract_dataset.append(abtract)
        keyphrases_dataset.append(new_keyphrases)
        
        
    return abstract_dataset , keyphrases_dataset

def masking_and_labeling_test(abstract_dataset , keyphrases_dataset, tokenizer):
        
    tokenized_texts = []
    formatted_labels_list = []

    for abstract, keyphrase_list in zip(abstract_dataset, keyphrases_dataset):
        tokenized_abstract = tokenizer.tokenize(abstract)
        tokenized_texts.append(tokenized_abstract)

        formatted_label = [0] * len(tokenized_abstract) 

        # marking the keyphrases
        for keyphrase in keyphrase_list:
            keyphrase_tokens = tokenizer.tokenize(keyphrase)
            for i in range(len(tokenized_abstract) - len(keyphrase_tokens) + 1):
                if tokenized_abstract[i:i + len(keyphrase_tokens)] == keyphrase_tokens:
                    for j in range(i, i + len(keyphrase_tokens)):
                        formatted_label[j] = 1  
    #                 break
        formatted_labels_list.append(formatted_label)
    
    return tokenized_texts , formatted_labels_list


def dataset_test(tokenized_texts , formatted_labels_list, tokenizer):
    max_length = max(len(txt) for txt in tokenized_texts)
    input_ids = torch.tensor([tokenizer.convert_tokens_to_ids(txt + ['[PAD]'] * (max_length - len(txt))) for txt in tokenized_texts])

    labels = torch.tensor([lbl + [0] * (max_length - len(lbl)) for lbl in formatted_labels_list])
    
    masks = torch.where(input_ids != 0, torch.tensor(1), torch.tensor(0))
    
    return input_ids , labels, masks


def return_keyphrase_test(sentence , logits_list , original_sequence , tokenizer , id_to_token):
    
    # converting the logit  into prediction sequnence
    predited_sequence = []
    batch_size = len(logits_list)
    
    for b in range(batch_size):
        
        batch_res = []
        logits_element = logits_list[b]
        
        for logit in logits_element:
        
            zero_class_logit = logit[0]
            one_class_logit = logit[1]

            if zero_class_logit > one_class_logit:
                batch_res.append(0)
            else:
                batch_res.append(1)
                
        predited_sequence.append(batch_res)
    
    # finding the predicted keyphrases
    predicted_keyphrases = []
    
    for b in range(batch_size):
        
        pred_seq = predited_sequence[b]
        sent = sentence[b]        
        sent_len = len(sent)
        
        pred_phrase = []        
        list_of_pred_phrase = []
        
        for index in range(sent_len):
            
            logit = pred_seq[index]
            word_index = sent[index]
            word = id_to_token[word_index]
            
            if logit == 0:
                
                if len(pred_phrase) < 1:
                    continue
                
                decoded_sentence = tokenizer.convert_tokens_to_string(pred_phrase)       
                list_of_pred_phrase.append(decoded_sentence)
                
                pred_phrase = []
            
            else:
                pred_phrase.append(word)
            
        if len(pred_phrase) > 0:
            decoded_sentence = tokenizer.convert_tokens_to_string(pred_phrase)       
            list_of_pred_phrase.append(decoded_sentence)
            
        predicted_keyphrases.append(list_of_pred_phrase)  
        
        
    # finding the original keyphrases
    original_keyphrases = []
    
    for b in range(batch_size):
        
        original_seq = original_sequence[b]
        sent = sentence[b]        
        sent_len = len(sent)
        
        original_phrase = []        
        list_of_original_phrase = []
        
        for index in range(sent_len):
            
            logit = original_seq[index]
            word_index = sent[index]
            word = id_to_token[word_index]
            
            if logit == 0:
                
                if len(original_phrase) < 1:
                    continue
                
                decoded_sentence = tokenizer.convert_tokens_to_string(original_phrase)       
                list_of_original_phrase.append(decoded_sentence)
                
                original_phrase = []
            
            else:
                original_phrase.append(word)
            
        if len(original_phrase) > 0:
            decoded_sentence = tokenizer.convert_tokens_to_string(original_phrase)       
            list_of_original_phrase.append(decoded_sentence)
            
        original_keyphrases.append(list_of_original_phrase)
    
    return predicted_keyphrases , original_keyphrases
    
def predict_keyphrases(model, tokenizer, abstract):
    model.eval()
    
    tokenized_abstract = tokenizer.tokenize(abstract)

    input_ids = tokenizer.convert_tokens_to_ids(tokenized_abstract)
    
    max_length = model.config.max_position_embeddings
    input_ids += [tokenizer.pad_token_id] * (max_length - len(input_ids))
    
    input_ids_tensor = torch.tensor(input_ids).unsqueeze(0).to(device)  
    
    with torch.no_grad():
        outputs = model(input_ids_tensor)
    
    predicted_labels = torch.argmax(outputs.logits, axis=-1).squeeze(0).tolist()
    
    predicted_keyphrases = []
    current_keyphrase = []
    
    for i, token_id in enumerate(input_ids):
        token = tokenizer.convert_ids_to_tokens(token_id)
        if token == '[PAD]':
            break
        if predicted_labels[i] == 1:
            current_keyphrase.append(token)
        else:
            if current_keyphrase:
                predicted_keyphrases.append(tokenizer.convert_tokens_to_string(current_keyphrase))
                current_keyphrase = []
    
    return predicted_keyphrases



def test(kpe_model_path, ner_model_path , nli_model_path , abtract_file_path):

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    
    best_model = BertForTokenClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=2, 
        output_attentions=False,
        output_hidden_states=False
    ).to(device)

    best_model.load_state_dict(torch.load(kpe_model_path))
    best_model.eval()

    abstract = read_input_file(abtract_file_path)
    
    predicted_keyphrases = list(set(predict_keyphrases(best_model, tokenizer, abstract)))


    class CustomDatasetInference(Dataset):
        def __init__(self, data, word2idx):
            self.data = data 
            self.word2idx = word2idx
            self.num_classes = 3 
            self.max_length = max(len(words) for words in data)

        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            words = self.data[idx]
            idx_list = [self.word2idx.get(word, 1) for word in words]  # Use 1 for 'unk' if word not in word2idx
            padded_idx_list = idx_list + [0] * (self.max_length - len(idx_list))
            return torch.tensor(padded_idx_list)
        

    class BiLSTMClassifier(nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            super(BiLSTMClassifier, self).__init__()
            self.embedding = nn.Embedding(input_size, hidden_size)
            self.bilstm = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=True)
            self.fc = nn.Linear(hidden_size * 2, num_classes)
            
        def forward(self, x):
            x = self.embedding(x)
            x, _ = self.bilstm(x)
            x = self.fc(x[:, -1, :])
            return x

    train_data = get_data(r'preprocessed_dataset\train\keyphrases')

    word2idx = dict()
    word2idx['pad'] = 0
    word2idx['unk'] = 1
    for text, label in train_data:
        for word in text:
            if word not in word2idx:
                word2idx[word] = len(word2idx)

    input_size = len(word2idx)
    hidden_size = 128
    num_classes = 3
    learning_rate = 0.001
    num_epochs = 10


    model = BiLSTMClassifier(input_size, hidden_size, num_classes).to(device)

    model.load_state_dict(torch.load(ner_model_path))
    predicted_labels = []

    preprocess_keyphrases = []

    for phrases in predicted_keyphrases:

        phrases = phrases.split()
        preprocess_text = []
        
        for word in phrases:
            preprocess_text.append(porter_stemmer.stem(word.lower()))
            
        preprocess_keyphrases.append(preprocess_text)

    inference_dataset = CustomDatasetInference(preprocess_keyphrases, word2idx)
    inference_dataloader = DataLoader(inference_dataset, batch_size=1, shuffle=False)

    with torch.no_grad():
        model.eval()

        for keyphrase in inference_dataloader:

            keyphrase  = keyphrase.to(device)
            output = model(keyphrase)

            _, predicted = torch.max(output, 1)

            if predicted.item() == 0:
                pred = "Process"

            elif predicted.item() == 1:
                pred = "Task"

            else:
                pred = "Material"

            predicted_labels.append(pred)    
    
    class InferenceDatasetNLI(Dataset):
        def __init__(self, keyphrases_list):
            self.keyphrases_list = keyphrases_list
            
            self.keyphrases_dataset = []

            self.generate_pairs()

        def __len__(self):
            return len(self.keyphrases_dataset)

        def __getitem__(self, idx):
            return self.keyphrases_dataset[idx]
        
        def pad_sequence_priv(self, sequence, max_length):
            padding_length = max_length - sequence.shape[1]
            if padding_length > 0:
                return torch.nn.functional.pad(sequence, (0, padding_length))
            else:
                return sequence

        def generate_pairs(self):
            max_seq_length = 15
            
            count_s = 0
            count_h = 0
            count_n = 0
            
            keyphrases_list = self.keyphrases_list
            
            for i in range(len(keyphrases_list)):
                for j in range(len(keyphrases_list)):
                    if i == j:
                        continue
                    tokenized_keyphrase_pair = tokenizer(self.keyphrases_list[i], self.keyphrases_list[j], padding=True, truncation=True, return_tensors="pt")

                    padded_input_ids = self.pad_sequence_priv(tokenized_keyphrase_pair['input_ids'], max_seq_length)
                    padded_attention_mask = self.pad_sequence_priv(tokenized_keyphrase_pair['attention_mask'], max_seq_length)
                    padded_token_type_ids = self.pad_sequence_priv(tokenized_keyphrase_pair['token_type_ids'], max_seq_length)

                    padded_tokenized_keyphrase_pair = {
                        'input_ids': padded_input_ids,
                        'attention_mask': padded_attention_mask,
                        'token_type_ids': padded_token_type_ids
                    }

                    self.keyphrases_dataset.append(padded_tokenized_keyphrase_pair)

    inference_dataset = InferenceDatasetNLI(predicted_keyphrases)
    inference_loader = DataLoader(inference_dataset, batch_size=1, shuffle=True)

    def pick_items(dictionary):
        keys = list(dictionary.keys())
        keys = random.sample(keys, 2)
        items = [[key, dictionary[key]] for key in keys]
        return items
    

    model_nli = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
    model_nli = model_nli.to(device)
    
    model_nli.load_state_dict(torch.load(nli_model_path))

    model_nli.eval()
    predictions = []
    

    true_labels = []
    output = dict()
    
    with torch.no_grad():
        for token_input in tqdm(inference_loader):
            
            input_ids = token_input['input_ids'].squeeze(1).to(device)
            attention_mask = token_input['attention_mask'].squeeze(1).to(device)  
            token_type_ids = token_input['token_type_ids'].squeeze(1).to(device) 

            outputs = model_nli(input_ids, attention_mask=attention_mask)
            _, predicted = torch.max(outputs.logits, 1)
            
            # Convert input_ids back to original words
            temp = input_ids.tolist()
            temp_words = [tokenizer.convert_ids_to_tokens(word) for word in temp]
            final_word_list = tokenizer.convert_tokens_to_string(temp_words[0]).split()
            
            first_sep_token_index =-1
            second_sep_token_index = -1
            
            for index in range(len(final_word_list)):
                word = final_word_list[index]
    #             print(word)
                
                if second_sep_token_index == -1 and first_sep_token_index == -1 and word == '[SEP]':
                    first_sep_token_index = index
                
                elif second_sep_token_index == -1 and first_sep_token_index != -1 and word == '[SEP]':
                    second_sep_token_index = index
            
            first_keyphrase = final_word_list[1:first_sep_token_index]
            second_keyphrase = final_word_list[first_sep_token_index+1 : second_sep_token_index]
            
    #         print(first_keyphrase)
    #         print(second_keyphrase)
            
            sentence1 = ' '.join(first_keyphrase)
            sentence2 = ' '.join(second_keyphrase)
            if predicted.item() == 1 or predicted.item() == 2:
                output[(sentence1, sentence2)] = predicted.item()

    output = pick_items(output)

    # for i in range(len(output)):
    #     if output[i][1] == 1:
    #         print(", ".join(output[i][0]),"-->", "Synonym")
    #     if output[i][1] == 2:
    #         print(", ".join(output[i][0]), "-->","Hyponym")

    
    with codecs.open("Result_file.txt" , "w" , "utf-8") as f :

        f.write("Abtract:  ")
        f.write(abstract)
        f.write("\n\n")
        f.write("Type       |         Keyphrases:     ")
        f.write("\n")


        for index in range(len(predicted_keyphrases)):

            keyphrase = predicted_keyphrases[index]
            type_keyphrase = predicted_labels[index]

            f.write(type_keyphrase +  "   |     " + keyphrase)
            f.write('\n')
        
        f.write("\n\n")
        f.write("NLI Results:-")
        f.write("\n")
        for i in range(len(output)):
            if output[i][1] == 1:
                f.write(", ".join(output[i][0]) + "      -->      " + "Synonym")
            if output[i][1] == 2:
                f.write(", ".join(output[i][0])+ "      -->      " + "Hyponym")
            
            f.write("\n")


#################################################################################################################################################
#####################################################       Main        #########################################################################
#################################################################################################################################################

def main(args):

    split = args[1][1]

    # Choosing the split whether it is train, eval, interface or invalid choise 
    if split == 't':
        
        if len(args) < 2 or  len(args) > 2:
            print("Not proper formart arguments for train!! \n\nCorrect command:-  'python project.py -t' ")
            return 
        
        train()


    elif split == 'e':
        if len(args) < 5 or len(args) > 5:
            print("Not proper formart arguments for Eval !! \n\nCorrect command:-  'python project.py -t keyphrase_model_path ner_model_path nli_model_path'")
            return 

        kpe_model_path = args[2]
        ner_model_path = args[3]
        nli_model_path = args[4]

        eval(kpe_model_path , ner_model_path, nli_model_path)
    
    elif split == 'i':

        if len(args) < 6 or len(args) > 6:
            print("Not proper formart arguments for Test !! \n\nCorrect command python project.py -t keyphrase_model_path ner_model_path nli_model_path abstract_file_path")
            return 

        kpe_model_path = args[2]
        ner_model_path = args[3]
        nli_model_path = args[4]
        abtract_file_path = args[5]
        test(kpe_model_path , ner_model_path ,nli_model_path, abtract_file_path)

    else:
        print("Read the README.md and choose valid operation!!!!")
        return 


if __name__ == "__main__":
    
    # Checking whether command is too short or not
    if len(sys.argv) < 2 :
        print("Please Read The README.md and Select the coorect command!!!!")
    
    else:
        main(sys.argv)
=======
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
>>>>>>> origin/main
