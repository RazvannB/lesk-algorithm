from eval import get_instance
import re
import itertools
import os
import re
import xml.etree.ElementTree as Et
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


# Constants
RELS = {
    "n": ["hyponyms", "part_meronyms", "hypernyms", "member_holonyms"],
    "a": ["similar_tos", "also_sees", "attributes"],
    "v": ["entailments", "causes", "hyponyms"]
}
POS = "n" # n, a, v
RELPAIRS = list(itertools.combinations(RELS[POS], 2))
STOPWORDS = stopwords.words('english')
STEMMER = PorterStemmer()
WINDOW = 3

CORPUS = "line"
FILENAME = "line-n.train.xml"

WORDNET_MAP = {
    'line': {
        'cord': ['line.n.18'],
        'division': ['line.n.29'],
        'formation': ['line.n.03', 'line.n.01'],
        'phone': ['telephone_line.n.02'],
        'product': ['line.n.22'],
        'text': ['note.n.02', 'line.n.05', 'line.n.27']
    },
    'hard': {
        'HARD1': ['difficult.a.01'],
        'HARD2': ['arduous.s.01'],
        'HARD3': ['hard.r.07']
    },
    'serve': {
        'SERVE10': ['serve.v.06'],
        'SERVE2': ['serve.v.01'],
        'SERVE6': ['service.v.01'],
        'SERVE12': ['serve.v.02'],
    },
    'interest': {
        'interest_1': ['interest.n.01'],
        'interest_2': ['interest.n.03'],
        'interest_3': ['pastime.n.01'],
        'interest_4': ['sake.n.01'],
        'interest_5': ['interest.n.05'],
        'interest_6': ['interest.n.04'],
    }
}


# Helper functions
def remove_non_alphanumeric(s):
    return re.sub(r'\W+ ', ' ', s).lower()


def cannot_be_added(s):
    # "neither the first nor the last word is a function word, that is a pronoun, preposition, article or conjunction"
    return s in STOPWORDS


def string_matching(first, second):
    result = []
    matched = ""
    current = ""
    first = remove_non_alphanumeric(first)
    first = " ".join([STEMMER.stem(word) for word in first.split()])
    second = remove_non_alphanumeric(second)
    second = " ".join([STEMMER.stem(word) for word in second.split()])

    for word in first.split():
        # check if first word can be added
        if current == "" and cannot_be_added(word):
            continue

        current += word

        if current in second:
            matched = current
            current += " "
        elif len(matched) > 0:
            # check if last word can be added
            while cannot_be_added(matched.split()[-1]):
                matched = " ".join(matched.split()[:-1])

            result.append(matched)
            second.replace(matched, "", 1)
            current = ""
            matched = ""

            # for in loop will lose current word
            if word in second and not cannot_be_added(word):
                matched = word
                current = word + " "
        else:
            current = ""
            
    return result


def no_of_words(s):
    return len(s.split())


def get_window_of_context(wordArg, phraseArg):
    phrase = phraseArg.split()
    target_index = phrase.index(wordArg)

    actual_window_left = WINDOW
    actual_window_right = WINDOW

    if target_index - WINDOW < 0:
        actual_window_left = target_index
        actual_window_right += abs(target_index - WINDOW)
    
    if target_index + actual_window_right >= len(phrase):
        actual_window_right = len(phrase) - target_index - 1

    if actual_window_right < WINDOW:
        diff = WINDOW - actual_window_right
        start_index = max(0, target_index - actual_window_left - diff)
        actual_window_left = target_index - start_index

    context = phrase[target_index - actual_window_left : target_index + actual_window_right + 1]
    return [word for word in context if word not in STOPWORDS]


def list_senses(context):
    return [word_senses(word) for word in context]


# Wordnet access
def synset(s, pos=POS):
    return wordnet.synsets(s, pos=pos)


def definitions(synsets):
    return " ".join([defin.definition() for defin in synsets])


def relation(rel, A):
    if rel == "hyponyms":
        return definitions(A.hyponyms())
    elif rel == "part_meronyms":
        return definitions(A.part_meronyms())
    elif rel == "hypernyms":
        return definitions(A.hypernyms())
    elif rel == "member_holonyms":
        return definitions(A.member_holonyms())
    elif rel == "similar_tos":
        return definitions(A.similar_tos())
    elif rel == "also_sees":
        return definitions(A.also_sees())
    elif rel == "attributes":
        return definitions(A.attributes())
    elif rel == "entailments":
        return definitions(A.entailments())
    elif rel == "causes":
        return definitions(A.causes())
    elif rel == "gloss":
        return [A.definition()]
    elif rel == "example":
        return " ".join(A.examples())
    else:
        return None


def word_senses(word):
    return synset(word)


# Lesk's algorithm (1986)
def score(definition1, definition2):
    result = 0
    for match in string_matching(definition1, definition2):
        result += no_of_words(match) ** 2
    return result


def relatedness(A, B):
    result = 0

    for pair in RELPAIRS:
        result += score(relation(pair[0], A), relation(pair[1], B))
        result += score(relation(pair[1], A), relation(pair[0], B))
    return result


def scoresense(word, phrase, crtSense):
    context = get_window_of_context(word, phrase)
    senses = list_senses(context)
    target_index = context.index(word)

    result = 0
    for idx, other_senses in enumerate(senses):
        if idx == target_index:
            continue

        for sense in other_senses:
            result += relatedness(crtSense, sense)

    return result


def predict(word, phrase, filter_function = None):

    if filter_function is None:
        filter_function = lambda x : True

    synsets = list(filter(filter_function, synset(word)))

    scores = [scoresense(word, phrase, sense) for sense in synsets]
    max_score = max(scores)
    max_idx = scores.index(max_score)

    print(synsets)

    winner = synsets[max_idx]

    return winner.name(), winner.definition()


def filter_eval(synset):
    for labelWN, synset_names in WORDNET_MAP[CORPUS].items():
        if synset.name() in synset_names:
            return True

    return False


def predict_eval(word, phrase):
    predSynset, _ = predict(word, phrase, filter_eval)
    print(predSynset)

    for labelWN, synset_names in WORDNET_MAP[CORPUS].items():
        if predSynset in synset_names:
            return labelWN

    return None


def main_eval():
    xml_file_name = FILENAME
    xml_file_path = os.path.join('data_eval', xml_file_name)

    # read xml
    tree = Et.parse(xml_file_path)
    root = tree.getroot()
    lexelt = root[0]
    
    instances = [get_instance(instance_el) for instance_el in lexelt]
    
    word = instances[1000]["target_word"]
    label = instances[1000]["senseid"]
    phrase = instances[1000]["text"]

    pred = predict_eval(word, phrase)
    print(word)
    print(phrase)
    print(pred) 
    print(label)



if __name__ == '__main__':
    main_eval()
