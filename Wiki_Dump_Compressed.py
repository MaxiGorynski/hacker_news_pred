from gensim.downloader import load

# Download and load the text8 corpus
dataset = load("text8")  # This will download it if missing

from gensim.models import Word2Vec

model = Word2Vec(dataset, vector_size=100, window=5, min_count=5, workers=4)
model.save("text8_word2vec.model")
print("✅ Model trained and saved from text8")

