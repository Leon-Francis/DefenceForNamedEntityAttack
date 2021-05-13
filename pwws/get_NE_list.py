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
    agnews_0 = {
        'PERSON': 'Williams',
        'NORP': 'European',
        'FAC': 'Olympic',
        'ORG': 'Microsoft',
        'GPE': 'Australia',
        'LOC': 'Earth',
        'PRODUCT': '#',
        'EVENT': 'Cup',
        'WORK_OF_ART': 'PowerBook',
        'LAW': 'Pacers-Pistons',
        'LANGUAGE': 'Chinese',
        'DATE': 'third-quarter',
        'TIME': 'Tonight',
        'MONEY': '#39;t',
        'QUANTITY': '#39;t',
        'ORDINAL': '11th',
        'CARDINAL': '1',
    }
    agnews_1 = {
        'PERSON': 'Bush',
        'NORP': 'Iraqi',
        'FAC': 'Outlook',
        'ORG': 'Microsoft',
        'GPE': 'Iraq',
        'LOC': 'Asia',
        'PRODUCT': '#',
        'EVENT': 'Series',
        'WORK_OF_ART': 'Nobel',
        'LAW': 'Constitution',
        'LANGUAGE': 'French',
        'DATE': 'third-quarter',
        'TIME': 'hours',
        'MONEY': '39;Keefe',
        'ORDINAL': '2nd',
        'CARDINAL': 'Two',
    }
    agnews_2 = {
        'PERSON': 'Arafat',
        'NORP': 'Iraqi',
        'FAC': 'Olympic',
        'ORG': 'AFP',
        'GPE': 'Baghdad',
        'LOC': 'Earth',
        'PRODUCT': 'Soyuz',
        'EVENT': 'Cup',
        'WORK_OF_ART': 'PowerBook',
        'LAW': 'Constitution',
        'LANGUAGE': 'Filipino',
        'DATE': 'Sunday',
        'TIME': 'evening',
        'MONEY': '39;m',
        'QUANTITY': '20km',
        'ORDINAL': 'eighth',
        'CARDINAL': '6',
    }
    agnews_3 = {
        'PERSON': 'Arafat',
        'NORP': 'Iraqi',
        'FAC': 'Olympic',
        'ORG': 'AFP',
        'GPE': 'Iraq',
        'LOC': 'Kashmir',
        'PRODUCT': 'Yukos',
        'EVENT': 'Cup',
        'WORK_OF_ART': 'Gazprom',
        'LAW': 'Pacers-Pistons',
        'LANGUAGE': 'Hebrew',
        'DATE': 'Saturday',
        'TIME': 'overnight',
        'MONEY': '39;m',
        'QUANTITY': '#39;t',
        'ORDINAL': '11th',
        'CARDINAL': '6',
    }
    agnews = [agnews_0, agnews_1, agnews_2, agnews_3]
    yahoo_0 = {
        'PERSON': 'Fantasy',
        'NORP': 'Russian',
        'FAC': 'Taxation',
        'ORG': 'Congress',
        'GPE': 'U.S.',
        'LOC': 'Sea',
        'PRODUCT': 'Variable',
        'EVENT': 'Series',
        'WORK_OF_ART': 'Stopping',
        'LAW': 'Constitution',
        'LANGUAGE': 'Hebrew',
        'DATE': '2004-05',
        'TIME': 'morning',
        'MONEY': '$ale',
        'QUANTITY': 'Hiberno-English',
        'ORDINAL': 'Tertiary',
        'CARDINAL': 'three',
    }
    yahoo_1 = {
        'PERSON': 'Equine',
        'NORP': 'Japanese',
        'FAC': 'Music',
        'ORG': 'Congress',
        'GPE': 'UK',
        'LOC': 'Sea',
        'PRODUCT': 'RuneScape',
        'EVENT': 'Series',
        'WORK_OF_ART': 'Stopping',
        'LAW': 'Strap-',
        'LANGUAGE': 'Spanish',
        'DATE': '2004-05',
        'TIME': 'night',
        'PERCENT': '100%',
        'MONEY': 'five-dollar',
        'QUANTITY': 'Hiberno-English',
        'ORDINAL': 'Sixth',
        'CARDINAL': '5',
    }
    yahoo_2 = {
        'PERSON': 'Equine',
        'NORP': 'Canadian',
        'FAC': 'Music',
        'ORG': 'Congress',
        'GPE': 'California',
        'LOC': 'Atlantic',
        'PRODUCT': 'Variable',
        'EVENT': 'Series',
        'WORK_OF_ART': 'Weight',
        'LANGUAGE': 'Filipino',
        'DATE': '2004-05',
        'TIME': 'night',
        'PERCENT': '100%',
        'MONEY': 'ten-dollar',
        'QUANTITY': '$ale',
        'ORDINAL': 'Tertiary',
        'CARDINAL': 'two',
    }
    yahoo_3 = {
        'PERSON': 'Equine',
        'NORP': 'Irish',
        'FAC': 'Music',
        'ORG': 'Congress',
        'GPE': 'California',
        'LOC': 'Sea',
        'PRODUCT': 'RuneScape',
        'EVENT': 'Series',
        'WORK_OF_ART': 'Weight',
        'LAW': 'Strap-',
        'LANGUAGE': 'Spanish',
        'DATE': '2004-05',
        'TIME': 'tonight',
        'PERCENT': '100%',
        'MONEY': 'five-dollar',
        'QUANTITY': 'Hiberno-English',
        'ORDINAL': 'Sixth',
        'CARDINAL': '5',
    }
    yahoo_4 = {
        'PERSON': 'Equine',
        'NORP': 'Irish',
        'FAC': 'Music',
        'ORG': 'Congress',
        'GPE': 'Canada',
        'LOC': 'Sea',
        'PRODUCT': 'Variable',
        'WORK_OF_ART': 'Stopping',
        'LAW': 'Constitution',
        'LANGUAGE': 'Spanish',
        'DATE': '2004-05',
        'TIME': 'seconds',
        'PERCENT': '100%',
        'MONEY': 'hundred-dollar',
        'QUANTITY': 'Hiberno-English',
        'ORDINAL': 'Tertiary',
        'CARDINAL': '100',
    }
    yahoo_5 = {
        'PERSON': 'Equine',
        'NORP': 'English',
        'FAC': 'Music',
        'ORG': 'Congress',
        'GPE': 'Australia',
        'LOC': 'Sea',
        'PRODUCT': 'Variable',
        'EVENT': 'Series',
        'WORK_OF_ART': 'Weight',
        'LAW': 'Strap-',
        'LANGUAGE': 'Filipino',
        'DATE': '2004-05',
        'TIME': 'seconds',
        'MONEY': 'hundred-dollar',
        'ORDINAL': 'Tertiary',
        'CARDINAL': '2000',
    }
    yahoo_6 = {
        'PERSON': 'Fantasy',
        'NORP': 'Islamic',
        'FAC': 'Music',
        'ORG': 'Congress',
        'GPE': 'California',
        'LOC': 'Sea',
        'PRODUCT': 'Variable',
        'EVENT': 'Series',
        'WORK_OF_ART': 'Stopping',
        'LANGUAGE': 'Filipino',
        'DATE': '2004-05',
        'TIME': 'seconds',
        'PERCENT': '100%',
        'MONEY': '$ale',
        'QUANTITY': '$ale',
        'ORDINAL': 'Tertiary',
        'CARDINAL': '100',
    }
    yahoo_7 = {
        'PERSON': 'Fantasy',
        'NORP': 'Canadian',
        'FAC': 'Music',
        'ORG': 'Congress',
        'GPE': 'UK',
        'LOC': 'West',
        'PRODUCT': 'Variable',
        'EVENT': 'Watergate',
        'WORK_OF_ART': 'Stopping',
        'LAW': 'Constitution',
        'LANGUAGE': 'Filipino',
        'DATE': '2004-05',
        'TIME': 'tonight',
        'PERCENT': '100%',
        'MONEY': '$ale',
        'QUANTITY': '$ale',
        'ORDINAL': 'Tertiary',
        'CARDINAL': '2000',
    }
    yahoo_8 = {
        'PERSON': 'Equine',
        'NORP': 'Japanese',
        'FAC': 'Music',
        'ORG': 'Congress',
        'GPE': 'Chicago',
        'LOC': 'Sea',
        'PRODUCT': 'Variable',
        'EVENT': 'Series',
        'WORK_OF_ART': 'Stopping',
        'LAW': 'Strap-',
        'LANGUAGE': 'Spanish',
        'DATE': '2004-05',
        'TIME': 'night',
        'PERCENT': '100%',
        'QUANTITY': '$ale',
        'ORDINAL': 'Sixth',
        'CARDINAL': '2',
    }
    yahoo_9 = {
        'PERSON': 'Equine',
        'NORP': 'Chinese',
        'FAC': 'Music',
        'ORG': 'Digital',
        'GPE': 'U.S.',
        'LOC': 'Atlantic',
        'PRODUCT': 'Variable',
        'EVENT': 'Series',
        'WORK_OF_ART': 'Weight',
        'LAW': 'Constitution',
        'LANGUAGE': 'Spanish',
        'DATE': '1918-1945',
        'TIME': 'night',
        'PERCENT': '100%',
        'MONEY': 'ten-dollar',
        'QUANTITY': 'Hiberno-English',
        'ORDINAL': 'Tertiary',
        'CARDINAL': '5'
    }
    yahoo = [
        yahoo_0, yahoo_1, yahoo_2, yahoo_3, yahoo_4, yahoo_5, yahoo_6, yahoo_7,
        yahoo_8, yahoo_9
    ]
    L = {'IMDB': imdb, 'AGNEWS': agnews, 'YAHOO': yahoo}


NE_list = NameEntityList()

if __name__ == '__main__':

    imdb_0 = defaultdict(list)
    imdb_1 = defaultdict(list)
    datas = []
    labels = []
    with open(r'./dataset/IMDB/aclImdb/train.std', 'r',
              encoding='utf-8') as file:
        for line in file:
            line = line.strip('\n')
            datas.append(line[:-1])
            labels.append(int(line[-1]))
    for idx, sen in enumerate(tqdm(datas)):
        doc = nlp(sen)
        for ent in doc.ents:
            if labels[idx] == 0:
                if ent.lower_ not in imdb_0[ent.label_]:
                    imdb_0[ent.label_].append(ent.lower_)
            else:
                if ent.lower_ not in imdb_1[ent.label_]:
                    imdb_1[ent.label_].append(ent.lower_)

    imdb_adv_0 = defaultdict(list)
    imdb_adv_1 = defaultdict(list)
    for key, value in imdb_0.items():
        for str in value:
            if str not in imdb_1[key]:
                imdb_adv_0[key].append(str)

    for key, value in imdb_1.items():
        for str in value:
            if str not in imdb_0[key]:
                imdb_adv_1[key].append(str)

    js_imdb_0 = json.dumps(imdb_adv_0)
    js_imdb_1 = json.dumps(imdb_adv_1)

    fileObject_0 = open('pwws/NE_dict/imdb_adv_1.json', 'w')
    fileObject_0.write(js_imdb_0)
    fileObject_0.close()

    fileObject_1 = open('pwws/NE_dict/imdb_adv_0.json', 'w')
    fileObject_1.write(js_imdb_1)
    fileObject_1.close()
