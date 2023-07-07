#####################################################################
# Text Helper
#
# To process text data
#
# __author__ = "Kornraphop Kawintiranon"
# __email__ = "kornraphop.k@gmail.com"
#
#####################################################################

import string
import tqdm
import re
import concurrent.futures
import multiprocessing
from nltk.corpus import stopwords
from nltk.tokenize.destructive import NLTKWordTokenizer
from emoji import demojize
from nltk.tokenize import TweetTokenizer


tokenizer = TweetTokenizer()

def normalize_token(token):
    lowercased_token = token.lower()
    if token.startswith("@"):
        return "@USER"
    elif lowercased_token.startswith("http") or lowercased_token.startswith("www"):
        return "HTTPURL"
    elif len(token) == 1:
        return demojize(token).replace(":[^: ]+:", " ").replace(" +", " ")
    else:
        if token == "’":
            return "'"
        elif token == "…":
            return "..."
        else:
            return token


def normalize_tweet(tweet):
    tweet = tweet.replace("’", "'")\
        .replace("…", "...")\
        .replace("🏻\u200d♂", "")\
        .replace("“|”", "''")\
        .replace("–|—", "-")

    tokens = tokenizer.tokenize(tweet)
    normTweet = " ".join([normalize_token(token) for token in tokens])

    normTweet = (
        normTweet.replace("cannot ", "can not ")
        .replace("n't ", " n't ")
        .replace("n 't ", " n't ")
        .replace("ca n't", "can't")
        .replace("ai n't", "ain't")
    )
    normTweet = (
        normTweet.replace("'m ", " 'm ")
        .replace("'re ", " 're ")
        .replace("'s ", " 's ")
        .replace("'ll ", " 'll ")
        .replace("'d ", " 'd ")
        .replace("'ve ", " 've ")
    )
    normTweet = (
        normTweet.replace(" p . m .", "  p.m.")
        .replace(" p . m ", " p.m ")
        .replace(" a . m .", " a.m.")
        .replace(" a . m ", " a.m ")
    )
    normTweet = (
        normTweet.replace("🇷 🇺", ":Russia:")
        .replace("🇺 🇦", ":Ukraine:")
        .replace("🇺 🇸", ":United_States:")
        .replace("🇺 🇳", ":United_Nations:")
        .replace("🇬 🇧", ":United_Kingdom:")
        .replace("🇹 🇷", ":Turkey:")
        .replace("🇵 🇰", ":Pakistan:")
    )

    return " ".join(normTweet.split())


def parallel_tokenize(corpus, tokenizer=None, n_jobs=-1):
    if tokenizer is None:
        tokenizer = NLTKWordTokenizer()
    if n_jobs < 0:
        n_jobs = multiprocessing.cpu_count() - 1
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) as executor:
        corpus_tokenized = list(
            tqdm.tqdm(executor.map(tokenizer.tokenize, corpus, chunksize=200), total=len(corpus), desc='Tokenizing')
        )
    to_remove = ["@user", "httpurl"]
    corpus_tokenized = [[token for token in tokens if token not in to_remove] for tokens in corpus_tokenized]
    return corpus_tokenized


def remove_stopwords(corpus, language='english'):
    stop_words = set(stopwords.words(language))
    processed_corpus = []
    for words in corpus:
        words = [w for w in words if not w in stop_words]
        processed_corpus.append(words)
    return processed_corpus


def remove_punctuations(corpus):
    punctuations = string.punctuation
    processed_corpus = []
    for words in corpus:
        words = [w for w in words if not w in punctuations]
        processed_corpus.append(words)
    return processed_corpus


def decontract(corpus):
    processed_corpus = []
    for phrase in tqdm.tqdm(corpus, desc="Decontracting"):
        phrase = re.sub(r"’", "\'", phrase)

        # specific
        phrase = re.sub(r"won\'t", "will not", phrase)
        phrase = re.sub(r"can\'t", "can not", phrase)

        # general
        phrase = re.sub(r"n\'t", " not", phrase)
        phrase = re.sub(r"\'re", " are", phrase)
        phrase = re.sub(r"\'s", " is", phrase)
        phrase = re.sub(r"\'d", " would", phrase)
        phrase = re.sub(r"\'ll", " will", phrase)
        phrase = re.sub(r"\'t", " not", phrase)
        phrase = re.sub(r"\'ve", " have", phrase)
        phrase = re.sub(r"\'m", " am", phrase)

        processed_corpus.append(phrase)
    return processed_corpus


def get_word_counts(corpus):
    # Initializing Dictionary
    d = {}

    # Counting number of times each word comes up in list of words (in dictionary)
    for words in tqdm.tqdm(corpus, desc="Word Counting"):
        for w in words:
            d[w] = d.get(w, 0) + 1
    return d
