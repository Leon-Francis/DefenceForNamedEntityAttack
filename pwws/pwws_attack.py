import __init__paths
import attr
import spacy
from nltk.corpus import wordnet as wn
from tools import str2seq, read_text_test_data
from config import config_device, config_pwws_use_NE, \
    config_data, config_dataset, model_path, config_pww_NNE_attack, BertConfig
import numpy as np
from get_NE_list import NE_list
from functools import partial
from torch import nn
import torch
import time
from transformers import BertTokenizer
from baseline_model import Baseline_Bert, Baseline_LSTM, Baseline_TextCNN
from baseline_config import Baseline_Config, dataset_config
from baseline_data import IMDB_Dataset
'''
    ATTENTION:
    Below three functions (PWWS, evaluate_word_saliency, adversarial_paraphrase)
    is an non official PyTorch version of https://github.com/JHL-HUST/PWWS
'''
nlp = spacy.load('en_core_web_sm')

supported_pos_tags = [
    'CC',  # coordinating conjunction, like "and but neither versus whether yet so"
    # 'CD',   # Cardinal number, like "mid-1890 34 forty-two million dozen"
    # 'DT',   # Determiner, like all "an both those"
    # 'EX',   # Existential there, like "there"
    # 'FW',   # Foreign word
    # 'IN',   # Preposition or subordinating conjunction, like "among below into"
    'JJ',  # Adjective, like "second ill-mannered"
    'JJR',  # Adjective, comparative, like "colder"
    'JJS',  # Adjective, superlative, like "cheapest"
    # 'LS',   # List item marker, like "A B C D"
    # 'MD',   # Modal, like "can must shouldn't"
    'NN',  # Noun, singular or mass
    'NNS',  # Noun, plural
    'NNP',  # Proper noun, singular
    'NNPS',  # Proper noun, plural
    # 'PDT',  # Predeterminer, like "all both many"
    # 'POS',  # Possessive ending, like "'s"
    # 'PRP',  # Personal pronoun, like "hers herself ours they theirs"
    # 'PRP$',  # Possessive pronoun, like "hers his mine ours"
    'RB',  # Adverb
    'RBR',  # Adverb, comparative, like "lower heavier"
    'RBS',  # Adverb, superlative, like "best biggest"
    # 'RP',   # Particle, like "board about across around"
    # 'SYM',  # Symbol
    # 'TO',   # to
    # 'UH',   # Interjection, like "wow goody"
    'VB',  # Verb, base form
    'VBD',  # Verb, past tense
    'VBG',  # Verb, gerund or present participle
    'VBN',  # Verb, past participle
    'VBP',  # Verb, non-3rd person singular present
    'VBZ',  # Verb, 3rd person singular present
    # 'WDT',  # Wh-determiner, like "that what whatever which whichever"
    # 'WP',   # Wh-pronoun, like "that who"
    # 'WP$',  # Possessive wh-pronoun, like "whose"
    # 'WRB',  # Wh-adverb, like "however wherever whenever"
]


def PWWS(
        doc,
        true_y,
        word_saliency_list=None,
        rank_fn=None,
        heuristic_fn=None,  # Defined in adversarial_tools.py
        halt_condition_fn=None,  # Defined in adversarial_tools.py
        verbose=True,
        sub_rate_limit=None):

    # defined in Eq.(8)
    def softmax(x):
        exp_x = np.exp(x)
        softmax_x = exp_x / np.sum(exp_x)
        return softmax_x

    heuristic_fn = heuristic_fn or (
        lambda _, candidate: candidate.similarity_rank)
    halt_condition_fn = halt_condition_fn or (lambda perturbed_text: False)
    perturbed_doc = doc
    perturbed_text = perturbed_doc.text

    substitute_count = 0  # calculate how many substitutions used in a doc
    substitute_tuple_list = []  # save the information of substitute word

    word_saliency_array = np.array(
        [word_tuple[2] for word_tuple in word_saliency_list])
    word_saliency_array = softmax(word_saliency_array)

    NE_candidates = NE_list.L[config_dataset][true_y]

    NE_tags = list(NE_candidates.keys())
    use_NE = config_pwws_use_NE  # whether use NE as a substitute
    NNE_attack = config_pww_NNE_attack

    max_len = config_data[config_dataset].padding_maxlen

    if sub_rate_limit:
        sub_rate_limit = int(sub_rate_limit * len(doc))
    else:
        sub_rate_limit = len(doc)

    # for each word w_i in x, use WordNet to build a synonym set L_i
    for (position, token, word_saliency, tag) in word_saliency_list:
        if position >= max_len:
            break

        candidates = []
        if use_NE:
            NER_tag = token.ent_type_
            if NER_tag in NE_tags:
                for idx, str in enumerate(NE_candidates[NER_tag]):
                    if idx >= 250:
                        break
                    candidate = SubstitutionCandidate(position, 0, token, str)
                    candidates.append(candidate)
            else:
                if NNE_attack:
                    candidates = _generate_synonym_candidates(
                        token=token, token_position=position, rank_fn=rank_fn)
        else:
            if NNE_attack:
                candidates = _generate_synonym_candidates(
                    token=token, token_position=position, rank_fn=rank_fn)

        if len(candidates) == 0:
            continue

        # The substitute word selection method R(w_i;L_i) defined in Eq.(4)
        sorted_candidates = zip(
            map(partial(heuristic_fn, doc.text), candidates), candidates)
        # Sorted according to the return value of heuristic_fn function, that is, \Delta P defined in Eq.(4)
        sorted_candidates = list(sorted(sorted_candidates, key=lambda t: t[0]))

        # delta_p_star is defined in Eq.(5); substitute is w_i^*
        delta_p_star, substitute = sorted_candidates.pop()

        # delta_p_star * word_saliency_array[position] equals H(x, x_i^*, w_i) defined in Eq.(7)
        substitute_tuple_list.append(
            (position, token.text, substitute,
             delta_p_star * word_saliency_array[position], token.tag_))

    # sort all the words w_i in x in descending order based on H(x, x_i^*, w_i)
    sorted_substitute_tuple_list = sorted(substitute_tuple_list,
                                          key=lambda t: t[3],
                                          reverse=True)

    # replace w_i in x^(i-1) with w_i^* to craft x^(i)
    # replace w_i in x^(i-1) with w_i^* to craft x^(i)
    NE_count = 0  # calculate how many NE used in a doc
    change_tuple_list = []
    for (position, token, substitute, score,
         tag) in sorted_substitute_tuple_list:
        if len(change_tuple_list) > sub_rate_limit:
            break
        # if score <= 0:
        #     break
        if nlp(token)[0].ent_type_ in NE_tags:
            NE_count += 1
        change_tuple_list.append((position, token, substitute, score, tag))
        perturbed_text = ' '.join(
            _compile_perturbed_tokens(perturbed_doc, [substitute]))
        perturbed_doc = nlp(perturbed_text)
        substitute_count += 1
        if halt_condition_fn(perturbed_text):
            if verbose:
                print("use", substitute_count, "substitution; use", NE_count,
                      'NE')
            sub_rate = substitute_count / len(doc)
            if substitute_count == 0:
                NE_rate = 0.0
            else:
                NE_rate = NE_count / substitute_count
            return perturbed_text, sub_rate, NE_rate, change_tuple_list

    if verbose:
        print("use", substitute_count, "substitution; use", NE_count, 'NE')
    sub_rate = substitute_count / len(doc)
    if substitute_count == 0:
        NE_rate = 0.0
    else:
        NE_rate = NE_count / substitute_count

    return perturbed_text, sub_rate, NE_rate, change_tuple_list


@attr.s
class SubstitutionCandidate:
    token_position = attr.ib()
    similarity_rank = attr.ib()
    original_token = attr.ib()
    candidate_word = attr.ib()


def vsm_similarity(doc, original, synonym):
    window_size = 3
    start = max(0, original.i - window_size)
    try:
        sim = doc[start:original.i + window_size].similarity(synonym)
    except:
        synonym = nlp(synonym.text)
        sim = doc[start:original.i + window_size].similarity(synonym)
    return sim


def _get_wordnet_pos(spacy_token):
    '''Wordnet POS tag'''
    pos = spacy_token.tag_[0].lower()
    if pos in ['r', 'n', 'v']:  # adv, noun, verb
        return pos
    elif pos == 'j':
        return 'a'  # adj


def _synonym_prefilter_fn(token, synonym):
    '''
    Similarity heuristics go here
    '''
    if (len(synonym.text.split()) > 2 or (  # the synonym produced is a phrase
            synonym.lemma == token.lemma)
            or (  # token and synonym are the same
                synonym.tag != token.tag)
            or (  # the pos of the token synonyms are different
                token.text.lower() == 'be')):  # token is be
        return False
    else:
        return True


def _generate_synonym_candidates(token, token_position, rank_fn=None):
    '''
    Generate synonym candidates.
    For each token in the doc, the list of WordNet synonyms is expanded.
    :return candidates, a list, whose type of element is <class '__main__.SubstitutionCandidate'>
            like SubstitutionCandidate(token_position=0, similarity_rank=10, original_token=Soft, candidate_word='subdued')
    '''
    if rank_fn is None:
        rank_fn = vsm_similarity
    candidates = []
    if token.tag_ in supported_pos_tags:
        wordnet_pos = _get_wordnet_pos(token)  # 'r', 'a', 'n', 'v' or None
        wordnet_synonyms = []

        synsets = wn.synsets(token.text, pos=wordnet_pos)
        for synset in synsets:
            wordnet_synonyms.extend(synset.lemmas())

        synonyms = []
        for wordnet_synonym in wordnet_synonyms:
            spacy_synonym = nlp(wordnet_synonym.name().replace('_', ' '))[0]
            synonyms.append(spacy_synonym)

        synonyms = filter(partial(_synonym_prefilter_fn, token), synonyms)

        candidate_set = set()
        for _, synonym in enumerate(synonyms):
            candidate_word = synonym.text
            if candidate_word in candidate_set:  # avoid repetition
                continue
            candidate_set.add(candidate_word)
            candidate = SubstitutionCandidate(token_position=token_position,
                                              similarity_rank=None,
                                              original_token=token,
                                              candidate_word=candidate_word)
            candidates.append(candidate)
    return candidates


def _compile_perturbed_tokens(doc, accepted_candidates):
    '''
    Traverse the list of accepted candidates and do the token substitutions.
    '''
    candidate_by_position = {}
    for candidate in accepted_candidates:
        candidate_by_position[candidate.token_position] = candidate

    final_tokens = []
    for position, token in enumerate(doc):
        word = token.text
        if position in candidate_by_position:
            candidate = candidate_by_position[position]
            word = candidate.candidate_word.replace('_', ' ')
        final_tokens.append(word)

    return final_tokens


def evaluate_word_saliency(doc, origin_vector, input_y, net):
    word_saliency_list = []

    # zero the code of the current word and calculate the amount of change in the classification probability
    max_len = config_data[config_dataset].padding_maxlen
    origin_prob = net.predict_prob(origin_vector, input_y)[0]
    for position in range(len(doc)):
        if position >= max_len:
            break
        without_word_vector = origin_vector.clone().detach().to(config_device)
        without_word_vector[position] = 0
        prob_without_word = net.predict_prob(without_word_vector, input_y)[0]

        # calculate S(x,w_i) defined in Eq.(6)
        word_saliency = origin_prob - prob_without_word
        word_saliency_list.append(
            (position, doc[position], word_saliency, doc[position].tag_))

    position_word_list = []
    for word in word_saliency_list:
        position_word_list.append((word[0], word[1]))

    return position_word_list, word_saliency_list


def adversarial_paraphrase(input_text,
                           origin_vector,
                           true_y,
                           net: nn.Module,
                           tokenizer,
                           verbose=True,
                           sub_rate_limit=None):
    '''
    Compute a perturbation, greedily choosing the synonym if it causes the most
    significant change in the classification probability after replacement
    :return perturbed_text

    : generated adversarial examples
    :return perturbed_y: predicted class of perturbed_text
    :return sub_rate: word replacement rate showed in Table 3
    :return change_tuple_list: list of substitute words
    '''
    def halt_condition_fn(perturbed_text):
        '''
        Halt if model output is changed.
        '''
        maxlen = config_data[config_dataset].padding_maxlen
        perturbed_vector = str2seq(perturbed_text, maxlen,
                                   tokenizer).to(config_device)
        predict = net.predict_class(perturbed_vector)[0]
        return predict != true_y

    def heuristic_fn(text, candidate):
        '''
        Return the difference between the classification probability of the original
        word and the candidate substitute synonym, which is defined in Eq.(4) and Eq.(5).
        '''
        doc = nlp(text)
        maxlen = config_data[config_dataset].padding_maxlen
        perturbed_tokens = _compile_perturbed_tokens(doc, [candidate])
        perturbed_doc = ' '.join(perturbed_tokens)
        perturbed_vector = str2seq(perturbed_doc, maxlen,
                                   tokenizer).to(config_device)
        adv_y = net.predict_prob(perturbed_vector, true_y)[0]
        ori_y = net.predict_prob(origin_vector, true_y)[0]

        return ori_y - adv_y

    doc = nlp(input_text)

    # PWWS
    position_word_list, word_saliency_list = evaluate_word_saliency(
        doc, origin_vector, true_y, net)
    perturbed_text, sub_rate, NE_rate, change_tuple_list = PWWS(
        doc,
        true_y,
        word_saliency_list=word_saliency_list,
        heuristic_fn=heuristic_fn,
        halt_condition_fn=halt_condition_fn,
        verbose=verbose,
        sub_rate_limit=sub_rate_limit)

    # print("perturbed_text after perturb_text:", perturbed_text)

    maxlen = config_data[config_dataset].padding_maxlen
    perturbed_vector = str2seq(perturbed_text, maxlen,
                               tokenizer).to(config_device)
    perturbed_y = net.predict_class(perturbed_vector)[0]
    if verbose:
        origin_prob = net.predict_prob(origin_vector, true_y)[0]
        perturbed_prob = net.predict_prob(perturbed_vector, true_y)[0]
        raw_score = origin_prob - perturbed_prob
        print('Prob before: ', origin_prob, '. Prob after: ', perturbed_prob,
              '. Prob shift: ', raw_score)
    return perturbed_text, perturbed_y, sub_rate, NE_rate, change_tuple_list


def get_fool_sentence_pwws(sentence: str, label: int, index: int, net,
                           tokenizer, verbose, sub_rate_limit):

    start = time.perf_counter()
    maxlen = config_data[config_dataset].padding_maxlen
    vector = str2seq(sentence, maxlen, tokenizer).to(config_device)
    label = torch.tensor(label).to(config_device)
    flag = -1
    predict = net.predict_class(vector)[0]
    end = -1
    if predict == label:
        sentence, adv_y, sub_rate, NE_rate, change_tuple_list = adversarial_paraphrase(
            sentence,
            vector,
            label,
            net,
            tokenizer,
            verbose,
            sub_rate_limit,
        )
        if adv_y != label:
            flag = 1
        else:
            flag = 0
        end = time.perf_counter() - start
    return sentence, flag, end


class BaselineTokenizer():
    def __init__(self):
        train_dataset_orig = IMDB_Dataset(train_data=True,
                                          if_mask_NE=False,
                                          if_replace_NE=False,
                                          if_attach_NE=True,
                                          debug_mode=False)
        self.vocab = train_dataset_orig.vocab
        self.tokenizer = train_dataset_orig.tokenizer

    def tokenize(self, sen):
        return self.tokenizer.tokenize(sen)

    def convert_tokens_to_ids(self, word):
        return self.vocab.get_index(word)


if __name__ == '__main__':
    # attempt_num = 100

    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # datas, labels = read_text_data(
    #     dataset_config[config_dataset].test_data_path, attempt_num)
    # baseline_model = Baseline_Bert(
    #     label_num=dataset_config[config_dataset].labels_num,
    #     linear_layer_num=BertConfig.linear_layer_num[config_dataset],
    #     dropout_rate=BertConfig.dropout_rate[config_dataset],
    #     is_fine_tuning=BertConfig.is_fine_tuning[config_dataset]).to(config_device)

    # if attach_NE:
    #     baseline_model.load_state_dict(
    #         torch.load(model_path[f'{config_dataset}_Bert_attach_NE'], map_location=config_device))
    # else:
    #     baseline_model.load_state_dict(
    #         torch.load(model_path[f'{config_dataset}_Bert'], map_location=config_device))

    # baseline_model.eval()
    # success_num = 0
    # try_all = 0
    # attack_time = 0
    # for idx, data in enumerate(datas):
    #     adv_s, flag, end = get_fool_sentence_pwws(data, labels[idx], idx,
    #                                               baseline_model, tokenizer,
    #                                               True, None)
    #     attack_time += end
    #     if flag == 1:
    #         success_num += 1
    #         try_all += 1
    #     elif flag == 0:
    #         try_all += 1
    # print(f'attempt_num:{attempt_num}')
    # print(f'attack_num:{try_all}')
    # print(f'attack_acc:{success_num / try_all}')
    # print(f'attack_time:{attack_time}')
    attempt_num = 100

    tokenizer = BaselineTokenizer()
    datas, labels = read_text_test_data(
        dataset_config[config_dataset].test_data_path, attempt_num)
    # baseline_model = Baseline_LSTM(num_hiddens=128,
    #                                num_layers=2,
    #                                word_dim=50,
    #                                vocab=tokenizer.vocab,
    #                                labels_num=2,
    #                                using_pretrained=False,
    #                                bid=False,
    #                                head_tail=False).to(config_device)

    baseline_model = Baseline_TextCNN(vocab=tokenizer.vocab,
                                      train_embedding_word_dim=50,
                                      is_static=True,
                                      using_pretrained=True,
                                      num_channels=[50, 50, 50],
                                      kernel_sizes=[3, 4, 5],
                                      labels_num=2,
                                      is_batch_normal=False).to(config_device)


    baseline_model.load_state_dict(
        torch.load(model_path[f'IMDB_TextCNN_attach_NE'], map_location=config_device))

    baseline_model.eval()
    success_num = 0
    try_all = 0
    attack_time = 0
    for idx, data in enumerate(datas):
        adv_s, flag, end = get_fool_sentence_pwws(data, labels[idx], idx,
                                                  baseline_model, tokenizer,
                                                  True, None)
        attack_time += end
        if flag == 1:
            success_num += 1
            try_all += 1
        elif flag == 0:
            try_all += 1
    print(f'attempt_num:{attempt_num}')
    print(f'attack_num:{try_all}')
    print(f'attack_acc:{success_num / try_all}')
    print(f'attack_time:{attack_time}')
