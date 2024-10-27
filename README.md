# Dataset
- Used Seam Eval - 2017

# Preprocessing 

- In preprocessing, it seperate out the keyphrase and it's type and it;s relation in 3 different type task
- It seperate out whole dataset in 3 different component :- train, test, validation
- Each folder contain 3 subfolder :- 
- 1. Keyphrase -  contain keyphrase and it's type
- 2. paragraph - contain paragraph of abstract
- 3. NLI - contain NLI sentences

## Command
```python
    python preprocessing.py
```

# Position Rank 

- It's unsupervised graph based keyphrases algorithm
- We weigh each candidate word with its inverse position in the document before any filters are applied. If the same word appears multiple times in the target document, then we sum all its position weights.

## Command

```python

    python .\main.py   SeamEval2017_docs_folder_path   SeamEval2017_gold_folder_path


    Example:-    python .\main.py .\data\data\SeamEval2017\docs\ .\data\data\SeamEval2017\gold\

```


# Transformers based models

- Transformer :- bert-based-uncased

## Task
1. Keyphrase extraction
2. Keyphrase NER task  (type of task  :- 1. Process 2. Method 3. Task)
3. Keyphrase NLI task  (Determine 2 keyphrase are Hypyonym or synonym or no relation)

1. For training :-
- It will finetune the bert model for 3 above mentioned task 

- Give the Result for Each task for training, Validation and testing

```python

    python main.py -t

```


2. For Evalution :-

- It test the model for above mentioned task for Validation and Test Dataset and give the result

```python

    python main.py  -e  keyphrase_exctration_model_path  ner_model_path  nli_model_path 

```

3. For Inference :-

- It will take the paragraph as input and print the result in result file
- Result file contain abstract, keyphrase and it's type and which 2 keyphrase are hyponym or synonym 

```python

    python main.py  -e  keyphrase_exctration_model_path  ner_model_path  nli_model_path  abstract_file_path

```



## Notes:-

- Preprocessed_should be in same folder where main.py is
- All model:- https://drive.google.com/drive/folders/1h-ncFwGeYtlP0wCdYhvKlu7OPqN_OjUS?usp=sharing