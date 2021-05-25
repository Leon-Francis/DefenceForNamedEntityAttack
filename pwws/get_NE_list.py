import copy
import spacy
from collections import Counter, defaultdict
from tqdm import tqdm
import json

labels_num = 2

nlp = spacy.load('en_core_web_sm')

NE_type_dict = {
    'PERSON': defaultdict(int),  # People, including fictional.
    # Nationalities or religious or political groups.
    'NORP': defaultdict(int),
    'FAC': defaultdict(int),  # Buildings, airports, highways, bridges, etc.
    'ORG': defaultdict(int),  # Companies, agencies, institutions, etc.
    'GPE': defaultdict(int),  # Countries, cities, states.
    # Non-GPE locations, mountain ranges, bodies of water.
    'LOC': defaultdict(int),
    'PRODUCT': defaultdict(int),  # Object, vehicles, foods, etc.(Not services)
    # Named hurricanes, battles, wars, sports events, etc.
    'EVENT': defaultdict(int),
    'WORK_OF_ART': defaultdict(int),  # Titles of books, songs, etc.
    'LAW': defaultdict(int),  # Named documents made into laws.
    'LANGUAGE': defaultdict(int),  # Any named language.
    'DATE': defaultdict(int),  # Absolute or relative dates or periods.
    'TIME': defaultdict(int),  # Times smaller than a day.
    'PERCENT': defaultdict(int),  # Percentage, including "%".
    'MONEY': defaultdict(int),  # Monetary values, including unit.
    'QUANTITY': defaultdict(int),  # Measurements, as of weight or distance.
    'ORDINAL': defaultdict(int),  # "first", "second", etc.
    # Numerals that do not fall under another type.
    'CARDINAL': defaultdict(int),
}


def recognize_named_entity(texts):
    '''
    Returns all NEs in the input texts and their corresponding types
    '''
    NE_freq_dict = copy.deepcopy(NE_type_dict)

    for text in texts:
        doc = nlp(text)
        for word in doc.ents:
            NE_freq_dict[word.label_][word.text] += 1
    return NE_freq_dict


class NameEntityList(object):
    # If the original input in IMDB belongs to class 0 (negative)
    f_0 = open('pwws/NE_dict/imdb_adv_0.json', 'r')
    content_0 = f_0.read()
    imdb_0 = json.loads(content_0)
    f_0.close()
    f_1 = open('pwws/NE_dict/imdb_adv_1.json', 'r')
    content_1 = f_1.read()
    imdb_1 = json.loads(content_1)
    f_1.close()
    imdb = [imdb_0, imdb_1]

    f_0 = open('pwws/NE_dict/SST2_adv_0.json', 'r')
    content_0 = f_0.read()
    sst2_0 = json.loads(content_0)
    f_0.close()
    f_1 = open('pwws/NE_dict/SST2_adv_1.json', 'r')
    content_1 = f_1.read()
    sst2_1 = json.loads(content_1)
    f_1.close()
    sst2 = [sst2_0, sst2_1]

    f_0 = open('pwws/NE_dict/AGNEWS_adv_0.json', 'r')
    content_0 = f_0.read()
    agnews_0 = json.loads(content_0)
    f_0.close()
    f_1 = open('pwws/NE_dict/AGNEWS_adv_1.json', 'r')
    content_1 = f_1.read()
    agnews_1 = json.loads(content_1)
    f_1.close()
    f_2 = open('pwws/NE_dict/AGNEWS_adv_2.json', 'r')
    content_2 = f_2.read()
    agnews_2 = json.loads(content_2)
    f_2.close()
    f_3 = open('pwws/NE_dict/AGNEWS_adv_3.json', 'r')
    content_3 = f_3.read()
    agnews_3 = json.loads(content_3)
    f_3.close()

    agnews = [agnews_0, agnews_1, agnews_2, agnews_3]

    L = {'IMDB': imdb, 'AGNEWS': agnews, 'SST2': sst2}


NE_list = NameEntityList()

if __name__ == '__main__':

    agnews_0 = defaultdict(list)
    agnews_1 = defaultdict(list)
    agnews_2 = defaultdict(list)
    agnews_3 = defaultdict(list)
    datas = []
    labels = []
    with open(r'./dataset/AGNEWS/train.std', 'r',
              encoding='utf-8') as file:
        for line in file:
            line = line.strip('\n')
            datas.append(line[:-1])
            labels.append(int(line[-1]))
    for idx, sen in enumerate(tqdm(datas)):
        doc = nlp(sen)
        for ent in doc.ents:
            if labels[idx] == 0:
                if ent.lower_ not in agnews_0[ent.label_]:
                    agnews_0[ent.label_].append(ent.lower_)
            elif labels[idx] == 1:
                if ent.lower_ not in agnews_1[ent.label_]:
                    agnews_1[ent.label_].append(ent.lower_)
            elif labels[idx] == 2:
                if ent.lower_ not in agnews_2[ent.label_]:
                    agnews_2[ent.label_].append(ent.lower_)
            elif labels[idx] == 3:
                if ent.lower_ not in agnews_3[ent.label_]:
                    agnews_3[ent.label_].append(ent.lower_)

    agnews_adv_0 = defaultdict(list)
    agnews_adv_1 = defaultdict(list)
    agnews_adv_2 = defaultdict(list)
    agnews_adv_3 = defaultdict(list)

    for key, value in agnews_1.items():
        for str in value:
            if str not in agnews_0[key]:
                agnews_adv_0[key].append(str)
    for key, value in agnews_2.items():
        for str in value:
            if str not in agnews_0[key]:
                agnews_adv_0[key].append(str)
    for key, value in agnews_3.items():
        for str in value:
            if str not in agnews_0[key]:
                agnews_adv_0[key].append(str)

    for key, value in agnews_0.items():
        for str in value:
            if str not in agnews_1[key]:
                agnews_adv_1[key].append(str)
    for key, value in agnews_2.items():
        for str in value:
            if str not in agnews_1[key]:
                agnews_adv_1[key].append(str)
    for key, value in agnews_3.items():
        for str in value:
            if str not in agnews_1[key]:
                agnews_adv_1[key].append(str)

    for key, value in agnews_0.items():
        for str in value:
            if str not in agnews_2[key]:
                agnews_adv_2[key].append(str)
    for key, value in agnews_1.items():
        for str in value:
            if str not in agnews_2[key]:
                agnews_adv_2[key].append(str)
    for key, value in agnews_3.items():
        for str in value:
            if str not in agnews_2[key]:
                agnews_adv_2[key].append(str)

    for key, value in agnews_0.items():
        for str in value:
            if str not in agnews_3[key]:
                agnews_adv_3[key].append(str)
    for key, value in agnews_1.items():
        for str in value:
            if str not in agnews_3[key]:
                agnews_adv_3[key].append(str)
    for key, value in agnews_2.items():
        for str in value:
            if str not in agnews_3[key]:
                agnews_adv_3[key].append(str)

    js_agnews_0 = json.dumps(agnews_adv_0)
    js_agnews_1 = json.dumps(agnews_adv_1)
    js_agnews_2 = json.dumps(agnews_adv_2)
    js_agnews_3 = json.dumps(agnews_adv_3)

    fileObject_0 = open('pwws/NE_dict/AGNEWS_adv_0.json', 'w')
    fileObject_0.write(js_agnews_0)
    fileObject_0.close()

    fileObject_1 = open('pwws/NE_dict/AGNEWS_adv_1.json', 'w')
    fileObject_1.write(js_agnews_1)
    fileObject_1.close()

    fileObject_2 = open('pwws/NE_dict/AGNEWS_adv_2.json', 'w')
    fileObject_2.write(js_agnews_2)
    fileObject_2.close()

    fileObject_3 = open('pwws/NE_dict/AGNEWS_adv_3.json', 'w')
    fileObject_3.write(js_agnews_3)
    fileObject_3.close()
