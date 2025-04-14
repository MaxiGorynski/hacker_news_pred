# Load the model
from gensim.models import Word2Vec
model = Word2Vec.load("text8_word2vec.model")

# Find words most similar to "king"
print(model.wv.most_similar("king"))

# How similar are "king" and "queen"?
print("Similarity between 'king' and 'queen':", model.wv.similarity("king", "queen"))

# Compare other pairs
print("Similarity between 'king' and 'apple':", model.wv.similarity("king", "apple"))


