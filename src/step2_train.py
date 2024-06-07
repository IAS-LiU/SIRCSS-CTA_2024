import networkx as nx
import probabilistic_word_embeddings as pwe
from probabilistic_word_embeddings.preprocessing import preprocess_standard, preprocess_partitioned
from probabilistic_word_embeddings.embeddings import LaplacianEmbedding
from probabilistic_word_embeddings.estimation import map_estimate
from probabilistic_word_embeddings.evaluation import evaluate_on_holdout_set
import json, itertools
import numpy as np

DIM = 100
LAMBDA1 = 200.0

with open('data.json') as f:
    data = json.load(f)

texts, vocabulary = data["texts"], data["vocabulary"]
g = nx.read_adjlist("prior.graph")

#print(len())
e = LaplacianEmbedding(vocabulary, graph=g, dimensionality=DIM, lambda1=LAMBDA1)
train_data = list(itertools.chain(*texts))
e = map_estimate(e, train_data, model="cbow", ws=5, epochs=2, batch_size=20000)

e.save("embedding-cbow.pkl")
#e.save("embedding.json")

