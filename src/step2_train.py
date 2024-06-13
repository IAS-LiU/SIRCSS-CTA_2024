import networkx as nx
import probabilistic_word_embeddings as pwe
from probabilistic_word_embeddings.preprocessing import preprocess_standard, preprocess_partitioned
from probabilistic_word_embeddings.embeddings import LaplacianEmbedding
from probabilistic_word_embeddings.estimation import map_estimate
from probabilistic_word_embeddings.evaluation import evaluate_on_holdout_set
import json, itertools
import numpy as np

# Specify hyperparameters
DIM = 100
LAMBDA1 = 200.0

# Load data
with open('data.json') as f:
    data = json.load(f)
texts, vocabulary = data["texts"], data["vocabulary"]

# Load prior graph
g = nx.read_adjlist("prior.graph")

# Define embedding. Use Laplacian embedding since we have an informative prior
e = LaplacianEmbedding(vocabulary, graph=g, dimensionality=DIM, lambda1=LAMBDA1)

# Concatenate the list of lists into one list
# i.e. [["the_2006", ..., "done_2006"], ... , ["yesterday_2012", ..., "followed_2012"]]
# into ["the_2006", ... "followed_2012"]
train_data = list(itertools.chain(*texts))

# Train / estimate embeddings using MAP estimation
# use model="sgns" if you want the SGNS variant
e = map_estimate(e, train_data, model="cbow", ws=5, epochs=2, batch_size=20000)

# Save resulting embedding
e.save("embedding-cbow.pkl")

