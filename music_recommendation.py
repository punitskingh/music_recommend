import pandas as pd

df = pd.read_csv(r"../../datasets/spotify_millsongdata.csv")

df.shape

df = df.sample(10000).drop("link", axis=1).reset_index(drop=True)

df.shape

# Text cleaning / Text preprocessing

df["text"] = (
    df["text"].str.lower().replace(r"^\W\s", " ").replace(r"\n", " ", regex=True)
)

import nltk
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()


def token(txt):
    token = nltk.word_tokenize(txt)
    a = [stemmer.stem(w) for w in token]
    return " ".join(a)


nltk.download("punkt_tab")

df["text"].apply(lambda x: token(x))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

tfid = TfidfVectorizer(analyzer="word", stop_words="english")

matrix = tfid.fit_transform(df["text"])

similar = cosine_similarity(matrix)

# Recommender Function


def recommender(song_name):
    idx = df[df["song"] == song_name].index[0]
    distance = sorted(list(enumerate(similar[idx])), reverse=True, key=lambda x: x[1])
    song = []
    for s_id in distance[1:20]:
        song.append(df.iloc[s_id[0]].song)

    return song


recommender("In Your Eyes")

import pickle

pickle.dump(similar, open("similarity.pkl", "wb"))

pickle.dump(df, open("df.pkl", "wb"))
