import networkx as nx
import probabilistic_word_embeddings as pwe
from probabilistic_word_embeddings.preprocessing import preprocess_standard, preprocess_partitioned
from probabilistic_word_embeddings.embeddings import LaplacianEmbedding
from probabilistic_word_embeddings.estimation import map_estimate
import pandas as pd
import numpy as np

DIM = 100
VAL_SPLIT = 0.2
YEAR0 = 2005
YEAR1 = 2012

df = pd.read_csv("articles_en.csv").head(10000)

df["year"] = df["Date"].str.split("/").str[0]
print(df)

df = df.dropna()

## Create train-val split
df["val"] = np.random.rand(len(df)) <= VAL_SPLIT
df_train = df[df["val"] == False]
df_val = df[df["val"] == True]

texts_train = [t.split() for t in df_train["Text"]]
labels_train = [y for y in df_train["year"]]
print(df)

## 
texts_train, vocab = preprocess_partitioned(texts_train, labels=labels_train, downsample=False)

#print(vocab)
traindata = []
for t in texts_train:
    traindata += t

print(traindata[:10])
print(traindata[-10:])



## Generate prior graph
g = nx.Graph()
years = range(YEAR0, YEAR1)

for year0, year1 in zip(years[:-1], years[1:]):
    for wd in set([wd.split("_")[0] for wd in vocabulary]):
        wd0 = f"{wd}_{year0}"
        wd1 = f"{wd}_{year1}"
        if wd0 in vocabulary and wd1 in vocabulary:
            g.add_edge(wd0, wd1)

print(list(g.edges)[:10])

## Grid search
for l1 in [50, 100, 250, 500, 1000]:
    e = LaplacianEmbedding(vocab, graph=g, dimensionality=100, lambda1=l1)
    e = map_estimate(e, traindata, epochs=5)
    hold
    print()
