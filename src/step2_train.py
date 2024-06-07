import networkx as nx
import probabilistic_word_embeddings as pwe
from probabilistic_word_embeddings.preprocessing import preprocess_standard, preprocess_partitioned
from probabilistic_word_embeddings.embeddings import LaplacianEmbedding
from probabilistic_word_embeddings.estimation import map_estimate
from probabilistic_word_embeddings.evaluation import evaluate_on_holdout_set
import pandas as pd
import numpy as np

DIM = 100
VAL_SPLIT = 0.2
YEAR0 = 2005
YEAR1 = 2012

df = pd.read_csv("articles_en.csv") #.head(100000)

# Extract year from the 'Date' column
df["year"] = df["Date"].str.split("/").str[0]
print(df)

# Drop null rows
df = df.dropna()

## Create train-val split
df["val"] = np.random.rand(len(df)) <= VAL_SPLIT

texts = [t.split() for t in df["Text"]]
labels = [y for y in df["year"]]
print(df)

## 
texts, vocab = preprocess_partitioned(texts, labels=labels, downsample=False)
print(len(vocab), "vocab size")
#print(vocab)
traindata, valdata = [], []
for t, is_validation in zip(texts, df["val"]):
    if is_validation:
        valdata += t
    else:
        traindata += t

print(traindata[:10])
print(traindata[-10:])


## Generate prior graph
g = nx.Graph()
years = range(YEAR0, YEAR1)

for year0, year1 in zip(years[:-1], years[1:]):
    for wd in set([wd.split("_")[0] for wd in vocab]):
        wd0 = f"{wd}_{year0}"
        wd1 = f"{wd}_{year1}"
        if wd0 in vocab and wd1 in vocab:
            g.add_edge(wd0, wd1)

print(list(g.edges)[:10])

best_val_likelihood = -10000
e_best = None

## Grid search
for l1 in [500, 50, 100, 250]:
    print("Prior strength: ", l1)
    e = LaplacianEmbedding(vocab, graph=g, dimensionality=100, lambda1=l1)
    e = map_estimate(e, traindata, epochs=15, batch_size=20000)
    val_likelihood = evaluate_on_holdout_set(e, valdata)

    print(val_likelihood)
    if val_likelihood > best_val_likelihood:
        print("BEst likelihood")
        best_val_likelihood = val_likelihood
        e_best = e

print(e_best)
