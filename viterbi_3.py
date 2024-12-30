"""
Part 4: Here should be your best version of viterbi, 
with enhancements such as dealing with suffixes/prefixes separately
"""
from collections import defaultdict
from math import log

epsilon_for_pt = 1e-5
emit_epsilon = 1e-5 

laplace_transition = 1e-5
laplace_emission = 1e-5

classes = ('NUM', 'TINY', 'SHORT_S', 'SHORT', 'LONG_S', 'LONG')

def word_type(word):
    if word[0].isdigit() and word[-1].isdigit(): 
        return 'NUM'
    if len(word) < 4: 
        return 'TINY'
    if len(word) < 10:
        return 'SHORT_S' if word[-1] == 's' else 'SHORT'
    return 'LONG_S' if word[-1] == 's' else 'LONG'

def training(sentences):
    """
    Computes initial tags, emission words and transition curr_tag-to-curr_tag probabilities
    :param sentences: [[(word1, tag2), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    :return: intitial curr_tag probs, emission words given curr_tag probs, transition of tags to tags probs
    """
    init_prob = defaultdict(lambda: 0)
    emit_prob = defaultdict(lambda: defaultdict(lambda: 0))
    trans_prob = defaultdict(lambda: defaultdict(lambda: 0))
    hapax_prolly_dict = defaultdict(lambda: defaultdict(lambda: 0))

    # TODO: (I)
    # Input the training set, output the formatted probabilities according to data statistics.
    count_tags_dict = defaultdict(lambda: 0)
    count_words_dict = defaultdict(lambda: 0)
    for sentence in sentences:
        for i in range(len(sentence) - 1):
            curr_word = sentence[i][0]
            curr_tag = sentence[i][1]
            next_tag = sentence[i+1][1]

            if curr_word not in count_words_dict: #wrd if not seen before
                count_words_dict[curr_word] = [0, curr_tag]
            
            count_words_dict[curr_word][0] += 1
            count_tags_dict[curr_tag] += 1
            emit_prob[curr_tag][curr_word] += 1
            trans_prob[curr_tag][next_tag] += 1
    
    tags = [curr_tag for curr_tag in list(emit_prob.keys())]
    for curr_tag in tags:
        init_prob[curr_tag] = count_tags_dict[curr_tag] / sum(count_tags_dict.values())
    for word in count_words_dict:
        if count_words_dict[word][0] == 1:
            hapax_prolly_dict[count_words_dict[word][1]][word_type(word)] += 1
    for curr_tag in tags:
        for class_type in classes:
            if hapax_prolly_dict[curr_tag][class_type] == 0:
                hapax_prolly_dict[curr_tag][class_type] = 1
    #normalize
    total = 0
    for curr_tag in hapax_prolly_dict:
        for class_type in classes:
            total += hapax_prolly_dict[curr_tag][class_type]
    
    for curr_tag in hapax_prolly_dict:
        for class_type in classes:
            hapax_prolly_dict[curr_tag][class_type] /= total
    
    for curr_tag in emit_prob:
        n = sum(emit_prob[curr_tag].values())
        V = len(emit_prob[curr_tag])
        
        for word in emit_prob[curr_tag]:
            numerator = emit_prob[curr_tag][word] + laplace_emission
            denominator = n + laplace_emission*V + laplace_emission
            emit_prob[curr_tag][word] = numerator/denominator
        
        for class_type in classes:
            h = hapax_prolly_dict[curr_tag][class_type]
            numerator = emit_prob[curr_tag][class_type] + h*laplace_emission
            denominator = n + h*laplace_emission*V + laplace_emission
            emit_prob[curr_tag][class_type] = numerator/denominator
    
    tag_len = len(tags)
    for tag1 in tags:
        nt = sum(trans_prob[tag1].values())
        
        for tag2 in tags:
            numerator = trans_prob[tag1][tag2] + laplace_transition
            denominator = nt + laplace_transition*tag_len
            trans_prob[tag1][tag2] = numerator/denominator
    
    return init_prob, emit_prob, trans_prob

def viterbi_stepforward(i, word, prev_prob, prev_predict_tag_seq, emit_prob, trans_prob):
    """
    Does one step of the viterbi function
    :param i: The i'th column of the lattice/MDP (0-indexing)
    :param word: The i'th observed word
    :param prev_prob: A dictionary of tags to probs representing the max probability of getting to each curr_tag laplace_transition in the
    previous column of the lattice
    :param prev_predict_tag_seq: A dictionary representing the predicted curr_tag sequences leading up to the previous column
    of the lattice for each curr_tag in the previous column
    :param emit_prob: Emission probabilities
    :param trans_prob: Transition probabilities
    :return: Current best log probs leading to the i'th column for each curr_tag, and the respective predicted curr_tag sequences
    """
    log_prob = {} # This should store the log_prob for all the tags at current column (i)
    predict_tag_seq = {} # This should store the tag sequence to reach each tag at column (i)
    tags = []

    for curr_tag in emit_prob.keys():
        tags.append(curr_tag)
    
    if i == 0:#for first word
        for curr_tag in tags:
            trans_prolly = log(trans_prob['START'][curr_tag])
            if word in emit_prob[curr_tag]:
                emit_prolly = log(emit_prob[curr_tag][word])
            else:
                emit_prolly = log(emit_prob[curr_tag][word_type(word)])
            log_prob[curr_tag] = prev_prob[curr_tag] + emit_prolly + trans_prolly
            predict_tag_seq[curr_tag] = [curr_tag]
    else:
        for curr_tag in tags:
            best_previous_tag = None
            prev_potential_prob = {}
            for prev_tag in prev_prob:
                if word in emit_prob[curr_tag]:
                    emit_prolly = log(emit_prob[curr_tag][word])
                else:
                    emit_prolly = log(emit_prob[curr_tag][word_type(word)])
                trans_prolly = log(trans_prob[prev_tag][curr_tag])                
                prev_potential_prob[prev_tag] = (prev_prob[prev_tag] + 
                                               emit_prolly + 
                                               trans_prolly)
            
            best_previous_tag = max(prev_potential_prob, key=prev_potential_prob.get)            
            log_prob[curr_tag] = prev_potential_prob[best_previous_tag]            
            predict_tag_seq[curr_tag] = prev_predict_tag_seq[best_previous_tag] + [curr_tag]
    return log_prob, predict_tag_seq

def viterbi_3(train, test):
    '''
    input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag2), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
    output: list of sentences, each sentence is a list of (word,curr_tag) pairs.
            E.g., [[(word1, tag2), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    init_prob, emit_prob, trans_prob = training(train)
    predicts = []
    
    for sen in range(len(test)):
        sentence=test[sen]
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
            log_prob, predict_tag_seq = viterbi_stepforward(i, sentence[i], log_prob, predict_tag_seq, emit_prob,trans_prob)
            
        # TODO:(III) 
        # according to the storage of probabilities and sequences, get the final prediction.

        best_tag_seq = predict_tag_seq[max(log_prob, key=log_prob.get)]
        tagged_sentence = [(sentence[i], best_tag_seq[i]) for i in range(len(sentence))]
        predicts.append(tagged_sentence)
        
    return predicts