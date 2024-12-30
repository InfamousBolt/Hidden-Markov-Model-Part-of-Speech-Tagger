"""
Part 3: Here you should improve viterbi to use better laplace smoothing for unseen words
This should do better than baseline and your first implementation of viterbi, especially on unseen words
Most of the code in this file is the same as that in viterbi_1.py
"""

from collections import defaultdict
from math import log

epsilon_for_pt = 1e-5
emit_epsilon = 1e-5

def handle_hapax(sentences):
    
    word_count_dict = defaultdict(int)
    word_tag_dict = defaultdict(lambda: defaultdict(int))
    hapax_tag_dict = defaultdict(int)
    hapax_prolly_dict = defaultdict(float)
    
    for s in sentences:
        for word, tag in s:
            word_count_dict[word] += 1
            word_tag_dict[word][tag] += 1
    hapax_count = 0
    
    
    for word, word_count in word_count_dict.items():
        if word_count == 1:
            for tag, tag_count in word_tag_dict[word].items():
                hapax_tag_dict[tag] += tag_count
                hapax_count += tag_count
    
   
    for s in sentences:
        for word, tag in s:
            if tag not in hapax_tag_dict:
                hapax_tag_dict[tag] = 1#giving 1 count to all hpx tags
                hapax_count += 1
    
    for tag, word_count in hapax_tag_dict.items():
        hapax_prolly_dict[tag] = word_count / hapax_count
    
    return hapax_prolly_dict

def training(sentences):
    """
    Computes initial tags, emission words and transition tag-to-tag probabilities
    :param sentences:
    :return: intitial tag probs, emission words given tag probs, transition of tags to tags probs
    """
    init_prob = defaultdict(lambda: 0) # {init tag: #}
    emit_prob = defaultdict(lambda: defaultdict(lambda: 0)) # {tag: {word: # }}
    trans_prob = defaultdict(lambda: defaultdict(lambda: 0)) # {tag0:{tag1: # }}
    hapax_prolly_dict = handle_hapax(sentences)

    # TODO: (I)
    # Input the training set, output the formatted probabilities according to data statistics.
    count_tags_dict = defaultdict(lambda: 0)
    count_words_dict = defaultdict(lambda: 0)
    for sentence in sentences:
        init_prob[sentence[1][1]] += 1
        for i in range(1, len(sentence) - 1):#excld START & END
            curr_word = sentence[i][0]
            curr_tag = sentence[i][1]
            next_tag = sentence[i+1][1]
            
            emit_prob[curr_tag][curr_word] += 1
            trans_prob[curr_tag][next_tag] += 1
            count_tags_dict[curr_tag] += 1
            count_words_dict[curr_word] += 1
    
    total_tags = len(count_tags_dict)
    total_sentences = len(sentences)
    
    for tag in init_prob:
        init_prob[tag] = (init_prob[tag] + epsilon_for_pt) / (total_sentences + epsilon_for_pt * total_tags + epsilon_for_pt)
    
    for tag in count_tags_dict:
        n = sum(emit_prob[tag].values())
        V = len(count_words_dict)
        scaled_e_epsilon = emit_epsilon * hapax_prolly_dict[tag]#new epsilon
        for word in count_words_dict:
            emit_prob[tag][word] = (emit_prob[tag][word] + scaled_e_epsilon) / (n + scaled_e_epsilon * V + scaled_e_epsilon)
        emit_prob[tag]['UNKNOWN'] = scaled_e_epsilon / (n + scaled_e_epsilon * V + scaled_e_epsilon)
    
    for tag1 in count_tags_dict:
        for tag2 in count_tags_dict:
            trans_prob[tag1][tag2] = (trans_prob[tag1][tag2] + epsilon_for_pt) / (count_tags_dict[tag1] + epsilon_for_pt * total_tags + epsilon_for_pt)
    
    return init_prob, emit_prob, trans_prob

def viterbi_stepforward(i, word, prev_prob, prev_predict_tag_seq, emit_prob, trans_prob):
    """
    Does one step of the viterbi function
    :param i: The i'th column of the lattice/MDP (0-indexing)
    :param word: The i'th observed word
    :param prev_prob: A dictionary of tags to probs representing the max probability of getting to each tag at in the
    previous column of the lattice
    :param prev_predict_tag_seq: A dictionary representing the predicted tag sequences leading up to the previous column
    of the lattice for each tag in the previous column
    :param emit_prob: Emission probabilities
    :param trans_prob: Transition probabilities
    :return: Current best log probs leading to the i'th column for each tag, and the respective predicted tag sequences
    """
    log_prob = {} # This should store the log_prob for all the tags at current column (i)
    predict_tag_seq = {} # This should store the tag sequence to reach each tag at column (i)
    
    # TODO: (II)
    # implement one step of trellis computation at column (i)
    # You should pay attention to the i=0 special case.
    for current_tag in emit_prob:
        if i == 0:#for first word
            if word in emit_prob[current_tag]:
                log_prob[current_tag] = prev_prob[current_tag] + log(emit_prob[current_tag][word])
            else:
                log_prob[current_tag] = prev_prob[current_tag] + log(emit_prob[current_tag]['UNKNOWN'])
            predict_tag_seq[current_tag] = [current_tag]
        else:
            best_previous_tag = None
            best_log_prob = float('-inf')
            for prev_tag in prev_prob:
                if word in emit_prob[current_tag]:
                    prob = prev_prob[prev_tag] + log(trans_prob[prev_tag][current_tag]) + log(emit_prob[current_tag][word])
                else:
                    prob = prev_prob[prev_tag] + log(trans_prob[prev_tag][current_tag]) + log(emit_prob[current_tag]['UNKNOWN'])
                
                if prob > best_log_prob:
                    best_log_prob = prob
                    best_previous_tag = prev_tag
            
            log_prob[current_tag] = best_log_prob
            predict_tag_seq[current_tag] = prev_predict_tag_seq[best_previous_tag] + [current_tag]
    return log_prob, predict_tag_seq

def viterbi_2(train, test):
    '''
    input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
    output: list of sentences, each s is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    init_prob, emit_prob, trans_prob = training(train)
    predicts = []
    
    for sen in range(len(test)):
        sentence = test[sen]
        length = len(sentence)
        log_prob = {}
        predict_tag_seq = {}
        # init log prob
        for t in emit_prob:
            if t in init_prob:
                log_prob[t] = log(init_prob[t])
            else:
                log_prob[t] = log(epsilon_for_pt)
            predict_tag_seq[t] = []
        
        # forward steps to calculate log probs for sentence
        for i in range(length):
            log_prob, predict_tag_seq = viterbi_stepforward(i, sentence[i], log_prob, predict_tag_seq, emit_prob, trans_prob)
        
        # TODO:(III) 
        # according to the storage of probabilities and sequences, get the final prediction.
        best_tag = max(log_prob, key=lambda tag: log_prob[tag])
        best_tag_seq = predict_tag_seq[best_tag]
        tagged_sentence = [(sentence[i], best_tag_seq[i]) for i in range(len(sentence))]
        predicts.append(tagged_sentence)
    
    return predicts