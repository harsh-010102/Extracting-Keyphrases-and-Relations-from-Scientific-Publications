
def firstRank(predicted, gold):
    """returns the the rank of the first correct predicted keyphrase"""

    length_of_predicted_keyphrase = len(predicted)
    length_of_gold_keyphrases = len(gold)


    index = 0

    while index < length_of_predicted_keyphrase:
        
        predicted_keword = predicted[index]

        if predicted_keword in gold:
            return index
        
        index = index + 1

    return 0


def Rprecision(predicted, gold, cut_off):

    predicted_set = set(predicted)
    gold_set = set(gold)

    predicted_len = len(predicted)*1.0
    gold_len = len(gold)*1.0

    hits = gold_set.intersection(predicted_set)

    hits_len = len(hits)*1.0

    r_precision = hits_len/cut_off if hits_len > 0.0 and predicted_len > 0.0 else 0.0

    return r_precision


def PRF(predicted , gold, cut_off):

    predicted = predicted[:cut_off]

    predicted_set = set(predicted)
    gold_set = set(gold)

    hits = gold_set.intersection(predicted_set)

    predicted_len = len(predicted)*1.0
    gold_len = len(gold)*1.0
    hits_len = len(hits)*1.0

    precision = hits_len/predicted_len if predicted_len>0.0 and hits_len >0.0 else 0.0
    recall = hits_len/gold_len if gold_len>0 and hits_len>0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0.0 else 0.0

    result_prf = dict()

    result_prf['precision'] = precision
    result_prf['recall'] = recall
    result_prf['f1-score'] = f1_score
    
    return result_prf



def PRF_range(predicted, gold, cut_off):

    precision_list = []
    recall_list = []
    f1_score_list = []

    index = 0

    while index < cut_off:
        
        temp_predicted = predicted[:index+1]

        temp_predicted_set = set(temp_predicted)
        gold_set = set(gold)

        hits = gold_set.intersection(temp_predicted_set)

        predicted_len = len(temp_predicted)
        gold_len = len(gold)
        hits_len = len(hits)

        precision = hits_len/predicted_len if predicted_len>0 and hits_len >0 else 0.0
        recall = hits_len/gold_len if gold_len>0 and hits_len>0 else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0.0 else 0.0


        precision_list.append(precision)
        recall_list.append(recall)
        f1_score_list.append(f1_score)

        index = index + 1

    return precision_list , recall_list , f1_score_list


def Bpref (predicted, gold):

    predicted_len = len(predicted)*1.0
    gold_len = len(gold)*1.0

    incorrect_prediction = 0
    correct_prediction = 0
    bpref = 0

    for pred in predicted:

        if pred not in gold:
            incorrect_prediction = incorrect_prediction + 1

        else: 
            temp = (incorrect_prediction*1.0)/predicted_len
            temp = 1 - temp
            bpref = bpref + temp
            correct_prediction = correct_prediction + 1


    final_bpref = (bpref*1.0)/correct_prediction if correct_prediction>0 else 0.0

    return final_bpref