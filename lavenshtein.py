import codecs
import operator
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# no xrange method on Python 3
def xrange(x):
    return iter(range(x))

def lev_dist(source, target):
    if source == target:
        return 0

    # Prepare a matrix
    slen, tlen = len(source), len(target)
    dist = [[0 for i in range(tlen+1)] for x in range(slen+1)]
    for i in xrange(slen+1):
        dist[i][0] = i
    for j in xrange(tlen+1):
        dist[0][j] = j

    # Counting distance, here is my function
    for i in xrange(slen):
        for j in xrange(tlen):
            cost = 0 if source[i] == target[j] else 1
            dist[i+1][j+1] = min(dist[i][j+1] + 1, # deletion
                            dist[i+1][j] + 1, # insertion
                            dist[i][j] + cost # substitution
                        )
    return dist[-1][-1]

def find_nearest_word(source, target):
    scores = {}
    for token in target:
        scores[token] = lev_dist(source, token)

    return sorted(scores.items(), key=operator.itemgetter(1))


# load words from a file into a list
def loadWords(file):
    sentences = []
    token_list = [] # create an empty list to hold the file contents
    contents = codecs.open(file, "r", "utf-8") # open the file
    for line in contents: # loop over the lines in the file
        line = re.sub( '\s+', ' ', line).strip()
        sentences.append(line)
        tokens = line.split(' ')
        for token in tokens:
            token_list.append(token.lower())
    return sentences, token_list

sentences, targets = loadWords('procedures.txt')
# # print(sentences)
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)
# corpus_tf_idf = tfidf_vectorizer.transform(['test alergi'])
# scores = cosine_similarity(corpus_tf_idf, tfidf_matrix)
# print('Test sentence \'test alergi\'')
# print('Max Score is '+ str(np.amax(scores)) +'. Procedure is '+ sentences[np.argmax(scores)])
while True:
    words = []
    var = input("Enter something: ")
    tokens = var.split(' ')
    for token in tokens:
        if not token in targets:
            words.append(find_nearest_word(token, targets)[0][0])
        else:
            words.append(token)
    corrected_words = ' '.join(words)
    corpus_tf_idf = tfidf_vectorizer.transform([corrected_words])
    scores = cosine_similarity(corpus_tf_idf, tfidf_matrix)
    sorted_scores = np.argsort(-scores)
    final_sentence = []
    for i in range(0,10):
        final_sentence.append(sentences[sorted_scores[0][i]] +'\n')
    final_sentence = [k for k in final_sentence if words[0] in k.lower()]
    print('Apakah yang anda maksud:\n '+ ' '.join(final_sentence))
    print('') # print new line
