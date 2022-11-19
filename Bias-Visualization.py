import random
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib import pyplot as plt
import json
import random
from numpy import linalg as LA
import scipy.sparse
from sklearn.decomposition import PCA 
import pandas as pd 
import re 
#
fname= 'bias_detection-main\data\w2v_gnews_small.txt'
# fname= 'C:\Users\Hp\Downloads\bias_detection-main\bias_detection-main\data\w2v_gnews_small.txt'

def load(fname):
    word = []
    vector = []
    with open (fname,"r") as file:
        for line in file:
            s = line.split()
            word.append(s[0])
            vector.append(s[1:])
    vector = np.array(vector,'float32')
    return vector,word 
vector, word =load(fname)

def v(word_):
    return vector[word.index(word_)]

target_word= 'he'
ents_vector =  v(target_word)
print("Word {0} corresponds to a vector {1}".format(target_word, ents_vector))

def diff(word1, word2):
    # she 
    v = np.subtract(vector[word.index(word1)], vector[word.index(word2)])
    return v/LA.norm(v)

word1= 'she'
word2= 'he'

gender_vec = diff(word1,word2)
print("gender vector = {0}".format(gender_vec))

word1= 'white'
word2= 'black'

racial_vec = diff(word1,word2)
print("racial vector = {0}".format(racial_vec))

name= 'data/professions.txt'
def load_prof(fname):
    word = []
    with open (fname,"r") as file:
        for line in file:
            s = line.split()
            word.append(s[0])
    return word
word_prof =load_prof(fname)


she_he_vec =sorted([(v(w).dot(gender_vec), w) for w in word_prof])


# Gender Bias in professions
print("Top male: ")
for i in range(0,21):
    print("she-he: {}".format(she_he_vec[i]))
print("\nTop Female: ")
for i in reversed(range(1, 21)):
    print("she-he: {} ".format(she_he_vec[-i]))


x = v('himself') - v('herself')
x = x / np.linalg.norm(x)
x.dot(gender_vec/np.linalg.norm(gender_vec))

    
def compute_neighbors_(thresh, max_words):
    thresh = float(thresh) 
    print("Computing neighbors")
    
    vecs = vector[:max_words]
    vecs=np.asarray(vecs)
    # Find similarity between words (cosine similarity)
    dots = vecs.dot(vecs.T)
    
    # create a sparse matrix choosing only those elements with a value above the threshold (similarity >0.5)
    dots = scipy.sparse.csr_matrix(dots * (dots >= 1-thresh/2))
    from collections import Counter
    rows, cols = dots.nonzero()
    nums = list(Counter(rows).values())
    
    # Print average num of values satisfying the similarity thresh for each word
    print("Mean:", np.mean(nums)-1)
    print("Median:", np.median(nums)-1)
    rows, cols, vecs = zip(*[(i, j, vecs[i]-vecs[j]) for i, j, x in zip(rows, cols, dots.data) if i<j])
    v_ = np.array([v/np.linalg.norm(v) for v in vecs])
    return rows, cols,v_



def best_analogies_dist_thresh(v,thresh=1, topn=100, max_words=50000): 
    vecs = vector[:max_words]
    vecs=np.asarray(vecs)
    vocab=word[:max_words]
    rows, cols, vecs = compute_neighbors_(thresh, max_words)

    # Find the cosine similarity of "vector of interest" with neighbors
    scores = vecs.dot(v/np.linalg.norm(v))
    pi = np.argsort(-abs(scores))
    ans = []
    for i in pi:
        row = rows[i] if scores[i] > 0 else cols[i]
        col = cols[i] if scores[i] > 0 else rows[i]
        ans.append((vocab[row], vocab[col], abs(scores[i])))
        if len(ans)==topn:
            break

    return ans


gender_check = best_analogies_dist_thresh(gender_vec, 1, 100, 50000)
gender_check


diff_vec = diff("he", "she")

gender_direction = sorted([(vector[word.index(w)].dot(diff_vec), w) for w in word_prof])

points = []
names = []

print("Top Female: ")
for i in range(0,20):
    points.append(gender_direction[i][0])
    names.append(gender_direction[i][1])
    print(gender_direction[i])
    
print("\nTop Male: ")
for i in range(1, 21):
    points.append(gender_direction[-i][0])
    names.append(gender_direction[-i][1])
    print(gender_direction[-i])


from adjustText import adjust_text

y = np.zeros(len(points))
fig, ax = plt.subplots()
ax.set_title('Professions along the Gender Scale', fontsize=16)
fig.set_size_inches(18, 5)
ax.set_xlim([-1, 1])
ax.set_xlabel("(she -> he)", fontsize=14)
ax.plot(points, y, '.')
plt.tick_params(axis='y', which='both', direction='inout', labelleft='off')
plt.rcParams.update({'font.size': 14})

texts = []
for x, y, s in zip(points, y, names):
    texts.append(plt.text(x, y, s))

adjust_text(texts)
plt.show()


# diff_vec = diff("rich", "poor")

# wealth_direction = sorted([(vector[word.index(w)].dot(diff_vec), w) for w in word_prof])

# points = []
# names = []

# print("Top Poor: ")
# for i in range(0,20):
#     points.append(wealth_direction[i][0])
#     names.append(wealth_direction[i][1])
#     print(wealth_direction[i])
    
# print("\nTop Rich: ")
# for i in range(1, 20):
#     points.append(wealth_direction[-i][0])
#     names.append(wealth_direction[-i][1])
#     print(wealth_direction[-i])


# from adjustText import adjust_text

# y = np.zeros(len(points))
# fig, ax = plt.subplots()
# ax.set_title('Professions along the Financial Scale', fontsize=16)
# fig.set_size_inches(18, 5)
# ax.set_xlim([-1, 1])
# ax.set_xlabel("(poor -> rich)", fontsize=14)
# ax.plot(points, y, '.')
# plt.tick_params(axis='y', which='both', direction='inout', labelleft='off')
# plt.rcParams.update({'font.size': 14})

# texts = []
# for x, y, s in zip(points, y, names):
#     texts.append(plt.text(x, y, s))

# adjust_text(texts)
# plt.show()

diff_vec = diff("black", "white")

race_direction = sorted([(vector[word.index(w)].dot(diff_vec), w) for w in word_prof])

points = []
names = []

print("Top white: ")
for i in range(0,20):
    points.append(race_direction[i][0])
    names.append(race_direction[i][1])
    print(race_direction[i])
    
print("\nTop black: ")
for i in range(1, 20):
    points.append(race_direction[-i][0])
    names.append(race_direction[-i][1])
    print(race_direction[-i])


# from adjustText import adjust_text

y = np.zeros(len(points))
fig, ax = plt.subplots()
ax.set_title('Professions along the Race Scale', fontsize=16)
fig.set_size_inches(18, 5)
ax.set_xlim([-1, 1])
ax.set_xlabel("(black -> white)", fontsize=14)
ax.plot(points, y, '.')
plt.tick_params(axis='y', which='both', direction='inout', labelleft='off')
plt.rcParams.update({'font.size': 14})

texts = []
for x, y, s in zip(points, y, names):
    texts.append(plt.text(x, y, s))

adjust_text(texts)
plt.show()


# diff_vec = diff("conservative", "liberal")

# party_direction = sorted([(vector[word.index(w)].dot(diff_vec), w) for w in word_prof])

# points = []
# names = []

# print("Top liberal: ")
# for i in range(0,20):
#     points.append(party_direction[i][0])
#     names.append(party_direction[i][1])
#     print(party_direction[i])
    
# print("\nTop conservative: ")
# for i in range(1, 20):
#     points.append(party_direction[-i][0])
#     names.append(party_direction[-i][1])
#     print(party_direction[-i])


# from adjustText import adjust_text

# y = np.zeros(len(points))
# fig, ax = plt.subplots()
# ax.set_title('Professions along the Party Scale', fontsize=16)
# fig.set_size_inches(18, 5)
# ax.set_xlim([-1, 1])
# ax.set_xlabel("(liberal -> conservative)", fontsize=14)
# ax.plot(points, y, '.')
# plt.tick_params(axis='y', which='both', direction='inout', labelleft='off')
# plt.rcParams.update({'font.size': 14})

# texts = []
# for x, y, s in zip(points, y, names):
#     texts.append(plt.text(x, y, s))

# adjust_text(texts)
# plt.show()


diff_vec =  diff("young", "old")

age_direction = sorted([(vector[word.index(w)].dot(diff_vec), w) for w in word_prof])

points = []
names = []

print("Top old: ")
for i in range(0,20):
    points.append(age_direction[i][0])
    names.append(age_direction[i][1])
    print(age_direction[i])
    
print("\nTop young: ")
for i in range(1, 20):
    points.append(age_direction[-i][0])
    names.append(age_direction[-i][1])
    print(age_direction[-i])


from adjustText import adjust_text

y = np.zeros(len(points))
fig, ax = plt.subplots()
ax.set_title('Professions along the Age Scale', fontsize=16)
fig.set_size_inches(18, 5)
ax.set_xlim([-1, 1])
ax.set_xlabel("(old -> young)", fontsize=14)
ax.plot(points, y, '.')
plt.tick_params(axis='y', which='both', direction='inout', labelleft='off')
plt.rcParams.update({'font.size': 14})

texts = []
for x, y, s in zip(points, y, names):
    texts.append(plt.text(x, y, s))

adjust_text(texts)
plt.show()

