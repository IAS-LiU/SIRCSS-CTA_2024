import networkx as nx
import probabilistic_word_embeddings as pwe
from probabilistic_word_embeddings.preprocessing import preprocess_standard, preprocess_partitioned
from probabilistic_word_embeddings.embeddings import LaplacianEmbedding
from probabilistic_word_embeddings.estimation import map_estimate
from probabilistic_word_embeddings.evaluation import evaluate_on_holdout_set
import pandas as pd
import numpy as np

df = pd.read_csv("https://github.com/ninpnin/siml-public/releases/download/xd/articles_en.csv")
df = df.dropna()
df["Year"] = df["Date"].str.split("/").str[0]
df = df.head(1000)

# Preprocess text
texts = [t.split() for t in df["Text"]]
labels = df["Year"]
texts, vocabulary = preprocess_partitioned(texts, labels, limit=20, downsample=False)
print(texts[0], texts[-1])

import json

with open('data.json', 'w') as f:
    data = {"texts": texts, "vocabulary": vocabulary}
    json.dump(data, f, indent=1, ensure_ascii=False)

# Create prior graph
years = sorted(list(set(df["Year"])))
g = nx.Graph()
for year0, year1 in zip(years[:-1], years[1:]):
    for wd in set([wd.split("_")[0] for wd in vocabulary]):
        wd0 = f"{wd}_{year0}"
        wd1 = f"{wd}_{year1}"
        if wd0 in vocabulary and wd1 in vocabulary:
            g.add_edge(wd0, wd1)
print(list(g.edges)[:10])
nx.write_adjlist(g, "prior.graph")