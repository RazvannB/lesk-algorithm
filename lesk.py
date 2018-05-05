import re
import itertools
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


# Constants
RELS = ["hyponym", "meronym"]
PHRASE = "a dog was barking at a tree in the park full of people"
MYWORD = "park"

RELPAIRS = list(itertools.combinations(RELS, 2))
POS = None # n, a, v
STOPWORDS = stopwords.words('english')
STEMMER = PorterStemmer()
WINDOW = 3


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

def target_word_index(context=None):
    if context == None:
        return PHRASE.split().index(MYWORD)
    else:
        return context.index(MYWORD)

def get_window_of_context():
    phraseArr = PHRASE.split()
    target_index = target_word_index()
    actual_window_left = WINDOW
    actual_window_right = WINDOW

    if target_index - WINDOW < 0:
        actual_window_left = target_index
        actual_window_right += abs(target_index - WINDOW)
    
    if target_index + actual_window_right >= len(phraseArr):
        actual_window_right = len(phraseArr) - target_index - 1

    if actual_window_right < WINDOW:
        diff = WINDOW - actual_window_right
        start_index = max(0, target_index - actual_window_left - diff)
        actual_window_left = target_index - start_index

    context = phraseArr[target_index - actual_window_left : target_index + actual_window_right + 1]
    return [word for word in context if word not in STOPWORDS]

def target_word(s):
    return s == MYWORD

def list_senses(context):
    return [word_senses(word) for word in context]


# Wordnet access
def synset(s, pos=POS):
    return wordnet.synsets(s, pos=pos)

def definitions(synsets):
    return " ".join([defin.definition() for defin in synsets])

def relation(rel, A):
    if rel == "hyponym":
        return definitions(A.hyponyms())
    elif rel == "hypernym":
        return definitions(A.hypernyms())
    elif rel == "meronym":
        return definitions(A.part_meronyms())
    elif rel == "holonym":
        return definitions(A.member_holonyms())
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

def scoresense(current_sense):
    context = get_window_of_context()
    senses = list_senses(context)
    target_index = target_word_index(context)

    result = 0
    for idx, other_senses in enumerate(senses):
        if idx == target_index:
            continue

        for sense in other_senses:
            result += relatedness(current_sense, sense)

    return result

def main():
    myword_synsets = synset(MYWORD)
    print([syn.name() for syn in myword_synsets])

    scores = [scoresense(sense) for sense in myword_synsets]
    max_score = max(scores)
    max_idx = scores.index(max_score)

    winner = myword_synsets[max_idx]
    print(winner.name())
    print(winner.definition())

main()
