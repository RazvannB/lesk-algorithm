from eval import get_instance
import re
import itertools
import os
import re
import xml.etree.ElementTree as Et
from random import shuffle
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from difflib import SequenceMatcher


# Constants
RELS = {
    "n": ["hyponyms", "part_meronyms", "hypernyms", "member_holonyms", "definition", "examples"],
    "a": ["similar_tos", "also_sees", "attributes", "definition", "example"],
    "v": ["entailments", "causes", "hyponyms", "definition", "example"]
}
POS = "n" # n, a, v
RELPAIRS = list(itertools.combinations(RELS[POS], 2)) + list(map(lambda rel: (rel, rel), RELS[POS]))
STOPWORDS = set(stopwords.words('english'))
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


def tokenize_and_preprocess(phrase):
    # Tokenize
    words = word_tokenize(phrase)
    # Remove punctuation
    words = [word for word in words if word.isalpha()]
    # Lowercase
    words = [word.lower() for word in words]
    # Remove stop words
    words = [word for word in words if not word in STOPWORDS]
    # Stem
    # words = [STEMMER.stem(word) for word in words]

    return words


def get_window_of_context(word, phrase):
    # word = STEMMER.stem(word)
    phrase_tokens = tokenize_and_preprocess(phrase)
    target_index = phrase_tokens.index(word)

    left_window = None
    right_window = None

    if target_index - WINDOW >= 0 and target_index + WINDOW < len(phrase_tokens):
        left_window = WINDOW
        right_window = WINDOW
    elif target_index - WINDOW < 0 and target_index + WINDOW >= len(phrase_tokens):
        left_window = target_index
        right_window = len(phrase_tokens) - target_index - 1
    elif target_index - WINDOW < 0:
        left_window = target_index
        right_window = WINDOW + (WINDOW - left_window)
    elif target_index + WINDOW >= len(phrase_tokens):
        right_window = len(phrase_tokens) - target_index - 1
        left_window = WINDOW - right_window

    assert left_window is not None and right_window is not None

    return phrase_tokens[target_index - left_window : target_index + right_window + 1]


def get_synsets_for_word(s, pos = None):
    if pos is None:
        return wordnet.synsets(s)
    return wordnet.synsets(s, pos=pos)


def combine_definitions(synsets):
    return " ".join([synset.definition() for synset in synsets])


def relation(rel, synset):
    if rel == "hyponyms":
        return combine_definitions(synset.hyponyms())
    elif rel == "part_meronyms":
        return combine_definitions(synset.part_meronyms())
    elif rel == "hypernyms":
        return combine_definitions(synset.hypernyms())
    elif rel == "member_holonyms":
        return combine_definitions(synset.member_holonyms())
    elif rel == "similar_tos":
        return combine_definitions(synset.similar_tos())
    elif rel == "also_sees":
        return combine_definitions(synset.also_sees())
    elif rel == "attributes":
        return combine_definitions(synset.attributes())
    elif rel == "entailments":
        return combine_definitions(synset.entailments())
    elif rel == "causes":
        return combine_definitions(synset.causes())
    elif rel == "definition":
        return synset.definition()
    elif rel == "examples":
        return " ".join(synset.examples())
    else:
        return None


def score(definition1, definition2):
    result = 0

    definition1_tokens = tokenize_and_preprocess(definition1)
    definition2_tokens = tokenize_and_preprocess(definition2)

    while True:
        sequence_matcher = SequenceMatcher(None, definition1_tokens, definition2_tokens)
        match_results = sequence_matcher.find_longest_match(0, len(definition1_tokens), 0, len(definition2_tokens))

        if match_results.size == 0:
            break
        
        result += match_results.size ** 2

        definition1_tokens = definition1_tokens[:(match_results.a)] + definition1_tokens[(match_results.a + match_results.size):]
        definition2_tokens = definition2_tokens[:(match_results.b)] + definition2_tokens[(match_results.b + match_results.size):]

    return result


def relatedness(synset_A, synset_B):
    relatedness_score = 0

    for (relation_0, relation_1) in RELPAIRS:
        relatedness_score += score(relation(relation_0, synset_A), relation(relation_1, synset_B))
        relatedness_score += score(relation(relation_1, synset_A), relation(relation_0, synset_B))

    return relatedness_score


def calculate_synset_score(word_target, phrase, synset_target):
    context = get_window_of_context(word_target, phrase)

    listOfSynsetLists = [get_synsets_for_word(word) for word in context if word_target != word]
    score = 0

    for synsets in listOfSynsetLists:
        for synset in synsets:
            score += relatedness(synset_target, synset)

    return score


def predict(word, phrase, filter_function = lambda x : True):

    synsets = list(filter(filter_function, get_synsets_for_word(word, POS)))

    scores = [calculate_synset_score(word, phrase, synset) for synset in synsets]
    # scores_dict = {synset.name(): calculate_synset_score(word, phrase, synset) for synset in synsets}
    
    max_score = max(scores)
    max_idx = scores.index(max_score)
    winner = synsets[max_idx]

    return winner.name(), winner.definition()


def filter_eval(synset):
    for labelWN, synset_names in WORDNET_MAP[CORPUS].items():
        if synset.name() in synset_names:
            return True

    return False


def predict_eval(word, phrase):
    print(phrase)

    synsets = list(filter(filter_eval, get_synsets_for_word(word, POS)))

    scores = [calculate_synset_score(word, phrase, synset) for synset in synsets]

    scores_per_category = {}
    for category, synsetNamesInCategory in WORDNET_MAP[CORPUS].items():
        scores_per_category[category] = 0

        for synset_idx in range(len(synsets)):
            if synsets[synset_idx].name() in synsetNamesInCategory:
                scores_per_category[category] += scores[synset_idx]
    
    winnerScore = -1000
    winnerCategory = None

    for categoryName, categoryScore in scores_per_category.items():
        if categoryScore > winnerScore:
            winnerScore = categoryScore
            winnerCategory = categoryName

    return winnerCategory


def main_eval():
    xml_file_name = FILENAME
    xml_file_path = os.path.join('data_eval', xml_file_name)

    # read xml
    tree = Et.parse(xml_file_path)
    root = tree.getroot()
    lexelt = root[0]
    
    limit = 3
    instances = [get_instance(instance_el) for instance_el in lexelt if get_instance(instance_el)['senseid'] == 'product']
    shuffle(instances)

    labels_pred = list(map(lambda instance: predict_eval(instance['target_word'], instance['text']), instances[:limit]))
    print("Predicted labels:\n", labels_pred)
    labels_true = list(map(lambda instance: instance['senseid'], instances[:limit]))
    print("True labels:\n", labels_true)
    
    
    print("Classification Report\n", classification_report(labels_true, labels_pred))
    print("Accuracy Score\n", accuracy_score(labels_true, labels_pred))
    print("\nConfusion Matrix\n", confusion_matrix(labels_true, labels_pred))


if __name__ == '__main__':
    main_eval()
