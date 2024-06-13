from probabilistic_word_embeddings.embeddings import LaplacianEmbedding
from probabilistic_word_embeddings.evaluation import nearest_neighbors
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

# Load embedding 
e = LaplacianEmbedding(saved_model_path="embedding-cbow.pkl")

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) /(np.linalg.norm(vec1) * np.linalg.norm(vec2))

# Calculate similarity
w1, w2 = "rap_2011", "pop_2011"
similarity = cosine_similarity(e[w1], e[w2])
print("Similarity", w1, w2, similarity)

## Plot similarity over time
w1, w2 = "cat", "dog"
rows = []
for year in range(2006, 2013):
  w1_year, w2_year = f"{w1}_{year}", f"{w2}_{year}"
  similarity = cosine_similarity(e[w1_year], e[w2_year])
  row = [year, similarity]
  rows.append(row)
df = pd.DataFrame(rows, columns=["year", "similarity"])
sns.lineplot(df, x="year", y="similarity")
plt.ylim(0, 1)

plt.savefig(f"similarity-{w1}-{w2}.png")


df = nearest_neighbors(e, [f"sports_{year}" for year in range(2006, 2013)], K=7)
print(df)

df = nearest_neighbors(e, [f"gay_{year}" for year in range(2006, 2013)], K=10)
print(df)