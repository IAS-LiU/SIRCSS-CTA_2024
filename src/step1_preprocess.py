import networkx as nx
from probabilistic_word_embeddings.preprocessing import preprocess_standard, preprocess_partitioned
import pandas as pd
import json

# Read in file from the URL
df = pd.read_csv("https://github.com/ninpnin/siml-public/releases/download/xd/articles_en.csv")
df = df.dropna()
df["Year"] = df["Date"].str.split("/").str[0] # Scrape date from YYYY/MM/DD format

# Run the preprocessing on a subset of the data first
# df = df.head(10000)
# Also, if you run out of memory, you can drop some of the data at this point

# Preprocess text
texts = [t.split() for t in df["Text"]]
labels = df["Year"]
texts, vocabulary = preprocess_partitioned(texts, labels, limit=20, downsample=False)

# Print out the first and last articles as a sanity check
print(texts[0], texts[-1])

# Save data as JSON
with open('data.json', 'w') as f:
    data = {"texts": texts, "vocabulary": vocabulary}
    json.dump(data, f, indent=1, ensure_ascii=False)

# Create prior graph
years = sorted(list(set(df["Year"])))
print(years)
g = nx.Graph()

# For each word, the vectors for consecutive years are connected
# eg. dog_2006 ~ dog_2007 and so on
for year0, year1 in zip(years[:-1], years[1:]):
    for wd in set([wd.split("_")[0] for wd in vocabulary]):
        wd0 = f"{wd}_{year0}"
        wd1 = f"{wd}_{year1}"

        # Only add the edge if both words are present in the data
        if wd0 in vocabulary and wd1 in vocabulary:
            g.add_edge(wd0, wd1)

# Print first 10 edges as a sanity check
print(list(g.edges)[:10])

# Save prior graph as an adjacency list
nx.write_adjlist(g, "prior.graph")
