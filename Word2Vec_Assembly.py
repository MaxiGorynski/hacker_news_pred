import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import re
from collections import Counter
import random
import os
from tqdm import tqdm


class Word2VecDataset(Dataset):
    def __init__(self, corpus_file, vocab, word_to_idx, window_size=5):
        self.corpus_file = corpus_file
        self.vocab = vocab
        self.word_to_idx = word_to_idx
        self.window_size = window_size

        # Count total number of valid training pairs for progress tracking
        self.total_pairs = 0
        with open(corpus_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Counting training pairs"):
                words = self._preprocess_text(line.strip())
                indices = [word_to_idx.get(word) for word in words]
                indices = [idx for idx in indices if idx is not None]

                for i in range(len(indices)):
                    window = random.randint(1, window_size)
                    start = max(0, i - window)
                    end = min(len(indices), i + window + 1)

                    # Count context words
                    context_indices = indices[start:i] + indices[i + 1:end]
                    self.total_pairs += len(context_indices)

        print(f"Total training pairs: {self.total_pairs}")

    def _preprocess_text(self, text):
        """Basic text preprocessing"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        return text.split()

    def __iter__(self):
        with open(self.corpus_file, 'r', encoding='utf-8') as f:
            for line in f:
                words = self._preprocess_text(line.strip())
                indices = [self.word_to_idx.get(word) for word in words]
                indices = [idx for idx in indices if idx is not None]

                for i, target_idx in enumerate(indices):
                    # Dynamic window size
                    window = random.randint(1, self.window_size)
                    start = max(0, i - window)
                    end = min(len(indices), i + window + 1)

                    # Get context words
                    context_indices = indices[start:i] + indices[i + 1:end]

                    # Skip if no context words
                    if not context_indices:
                        continue

                    # Yield target-context pairs
                    for context_idx in context_indices:
                        yield target_idx, context_idx

    def __len__(self):
        return self.total_pairs


class NegativeSamplingDataset(Dataset):
    def __init__(self, word2vec_dataset, vocab_size, negative_samples=5):
        self.word2vec_dataset = word2vec_dataset
        self.vocab_size = vocab_size
        self.negative_samples = negative_samples

        # Create negative sampling table
        self.negative_table = self._create_negative_table()
        self.table_size = len(self.negative_table)

    def _create_negative_table(self, table_size=100000000):
        """Create table for negative sampling"""
        # Count frequency of each word in vocabulary
        word_counts = Counter()
        with open(self.word2vec_dataset.corpus_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Building negative sampling table"):
                words = self.word2vec_dataset._preprocess_text(line.strip())
                words = [word for word in words if word in self.word2vec_dataset.word_to_idx]
                word_counts.update(words)

        # Convert counts to probabilities with power of 0.75
        word_freqs = np.zeros(self.vocab_size)
        for word, count in word_counts.items():
            if word in self.word2vec_dataset.word_to_idx:
                idx = self.word2vec_dataset.word_to_idx[word]
                word_freqs[idx] = count

        word_freqs = np.power(word_freqs, 0.75)
        word_freqs = word_freqs / np.sum(word_freqs)

        # Create table where frequency determines how many times a word is added
        negative_table = []
        for idx, freq in enumerate(word_freqs):
            negative_table.extend([idx] * int(freq * table_size))

        # Make sure table is exactly table_size
        if len(negative_table) > table_size:
            negative_table = negative_table[:table_size]
        else:
            # Pad with random indices if necessary
            padding = table_size - len(negative_table)
            negative_table.extend(np.random.randint(0, self.vocab_size, padding).tolist())

        return np.array(negative_table)

    def __iter__(self):
        for target_idx, context_idx in self.word2vec_dataset:
            # Sample negative indices
            neg_indices = np.random.choice(self.negative_table, self.negative_samples)

            # Make sure target and context are not in negative samples
            for i, neg_idx in enumerate(neg_indices):
                if neg_idx == target_idx or neg_idx == context_idx:
                    neg_indices[i] = (neg_idx + 1) % self.vocab_size

            yield target_idx, context_idx, neg_indices

    def __len__(self):
        return len(self.word2vec_dataset)


class SkipGramNegativeSampling(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGramNegativeSampling, self).__init__()

        # Embedding matrices
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # Initialize weights as in original Word2Vec
        self.word_embeddings.weight.data.uniform_(-0.5 / embedding_dim, 0.5 / embedding_dim)
        self.context_embeddings.weight.data.zero_()

    def forward(self, target_word, context_word, negative_samples):
        # Get embeddings
        word_emb = self.word_embeddings(target_word)  # [batch_size, embed_dim]
        context_emb = self.context_embeddings(context_word)  # [batch_size, embed_dim]
        neg_embs = self.context_embeddings(negative_samples)  # [batch_size, neg_samples, embed_dim]

        # Positive score: dot product between word and context embeddings
        pos_score = torch.sum(word_emb * context_emb, dim=1)  # [batch_size]
        pos_score = torch.clamp(pos_score, max=10, min=-10)
        pos_loss = -torch.mean(nn.functional.logsigmoid(pos_score))

        # Negative score: dot product between word and negative sample embeddings
        neg_score = torch.bmm(neg_embs, word_emb.unsqueeze(2)).squeeze()  # [batch_size, neg_samples]
        neg_score = torch.clamp(neg_score, max=10, min=-10)
        neg_loss = -torch.mean(nn.functional.logsigmoid(-neg_score))

        return pos_loss + neg_loss

    def get_word_embeddings(self):
        return self.word_embeddings.weight.data.cpu().numpy()


class Word2Vec:
    def __init__(self, vector_size=300, window_size=5, min_count=5, negative_samples=5,
                 learning_rate=0.025, epochs=5, batch_size=512):
        self.vector_size = vector_size
        self.window_size = window_size
        self.min_count = min_count
        self.negative_samples = negative_samples
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size

        self.word_to_idx = {}
        self.idx_to_word = {}
        self.vocab = []
        self.model = None

    def _preprocess_text(self, text):
        """Basic text preprocessing"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        return text.split()

    def build_vocabulary(self, corpus_file):
        """Build vocabulary from corpus file"""
        print("Building vocabulary...")
        word_counts = Counter()

        # First pass: count words
        with open(corpus_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(tqdm(f, desc="Counting words")):
                words = self._preprocess_text(line.strip())
                word_counts.update(words)

        # Filter by minimum count
        self.vocab = [word for word, count in word_counts.items()
                      if count >= self.min_count]

        # Create word-to-index mappings
        for i, word in enumerate(self.vocab):
            self.word_to_idx[word] = i
            self.idx_to_word[i] = word

        print(f"Vocabulary size: {len(self.vocab)}")

    def train(self, corpus_file, use_cuda=False):
        """Train the Word2Vec model using skip-gram with negative sampling"""
        if not self.vocab:
            self.build_vocabulary(corpus_file)

        # Set device
        device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # Create dataset and dataloader
        dataset = Word2VecDataset(corpus_file, self.vocab, self.word_to_idx, self.window_size)
        neg_dataset = NegativeSamplingDataset(dataset, len(self.vocab), self.negative_samples)

        # Create model
        self.model = SkipGramNegativeSampling(len(self.vocab), self.vector_size).to(device)
        optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

        print("Starting training...")
        for epoch in range(self.epochs):
            total_loss = 0
            batch_count = 0

            # Reset dataset iterator
            dataloader = iter(neg_dataset)

            # Create progress bar
            progress_bar = tqdm(total=len(neg_dataset), desc=f"Epoch {epoch + 1}/{self.epochs}")

            # Batch processing
            target_batch = []
            context_batch = []
            neg_batch = []

            for target_idx, context_idx, neg_indices in dataloader:
                target_batch.append(target_idx)
                context_batch.append(context_idx)
                neg_batch.append(neg_indices)

                # Process in batches
                if len(target_batch) >= self.batch_size:
                    # Convert to tensors
                    targets = torch.LongTensor(target_batch).to(device)
                    contexts = torch.LongTensor(context_batch).to(device)
                    negatives = torch.LongTensor(neg_batch).to(device)

                    # Forward and backward pass
                    optimizer.zero_grad()
                    loss = self.model(targets, contexts, negatives)
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    batch_count += 1

                    # Update progress and reset batch
                    progress_bar.update(len(target_batch))
                    progress_bar.set_postfix({"Loss": f"{total_loss / batch_count:.4f}"})

                    target_batch = []
                    context_batch = []
                    neg_batch = []

            # Process any remaining items
            if target_batch:
                targets = torch.LongTensor(target_batch).to(device)
                contexts = torch.LongTensor(context_batch).to(device)
                negatives = torch.LongTensor(neg_batch).to(device)

                optimizer.zero_grad()
                loss = self.model(targets, contexts, negatives)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                batch_count += 1

                progress_bar.update(len(target_batch))

            progress_bar.close()

            # Update learning rate
            scheduler.step()

            print(f"Epoch {epoch + 1} completed, Avg. Loss: {total_loss / batch_count:.4f}")

    def save(self, filename):
        """Save the trained word vectors"""
        if self.model is None:
            raise ValueError("Model not trained yet")

        # Get word embeddings
        embeddings = self.model.get_word_embeddings()

        # Save in word2vec format
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"{len(self.vocab)} {self.vector_size}\n")
            for word, idx in self.word_to_idx.items():
                vector_str = ' '.join(str(val) for val in embeddings[idx])
                f.write(f"{word} {vector_str}\n")

    def load(self, filename):
        """Load pre-trained word vectors"""
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.vocab = []

        with open(filename, 'r', encoding='utf-8') as f:
            header = f.readline().split()
            vocab_size = int(header[0])
            self.vector_size = int(header[1])

            # Initialize model
            self.model = SkipGramNegativeSampling(vocab_size, self.vector_size)
            embeddings = np.zeros((vocab_size, self.vector_size))

            for i, line in enumerate(f):
                parts = line.rstrip().split(' ')
                word = parts[0]

                self.word_to_idx[word] = i
                self.idx_to_word[i] = word
                self.vocab.append(word)

                embeddings[i] = np.array([float(x) for x in parts[1:]])

            # Set embeddings
            self.model.word_embeddings.weight.data = torch.FloatTensor(embeddings)

    def get_vector(self, word):
        """Get vector for a word"""
        if self.model is None:
            raise ValueError("Model not trained yet")

        if word in self.word_to_idx:
            idx = self.word_to_idx[word]
            return self.model.get_word_embeddings()[idx]
        return None

    def get_most_similar(self, word, n=10):
        """Find most similar words"""
        if self.model is None:
            raise ValueError("Model not trained yet")

        if word not in self.word_to_idx:
            return []

        # Get word vector
        word_idx = self.word_to_idx[word]
        word_vec = self.model.get_word_embeddings()[word_idx]

        # Compute similarities with all other words
        embeddings = self.model.get_word_embeddings()

        # Normalize for cosine similarity
        norms = np.sqrt(np.sum(embeddings ** 2, axis=1))
        normalized_embeddings = embeddings / norms[:, np.newaxis]

        word_vec_norm = word_vec / np.linalg.norm(word_vec)
        similarities = np.dot(normalized_embeddings, word_vec_norm)

        # Get most similar words
        most_similar = []
        indices = np.argsort(-similarities)

        for idx in indices[:n + 1]:
            if idx != word_idx:
                most_similar.append((self.idx_to_word[idx], similarities[idx]))
                if len(most_similar) >= n:
                    break

        return most_similar


# Wikipedia processor for creating training data
def process_wikipedia(wiki_dump_path, output_fil  e, max_articles=None):
    """Process Wikipedia dump to text file for Word2Vec training"""
    import bz2
    from xml.etree import ElementTree as ET

    def extract_text(elem):
        """Extract text from a Wikipedia page element"""
        title = None
        text = None
        redirect = False
        namespace = '0'  # Default to main namespace

        for child in elem:
            if child.tag.endswith('title'):
                title = child.text
            elif child.tag.endswith('redirect'):
                redirect = True
            elif child.tag.endswith('ns'):
                namespace = child.text
            elif child.tag.endswith('revision'):
                for rc in child:
                    if rc.tag.endswith('text'):
                        text = rc.text

        return title, text, redirect, namespace

    def clean_wiki_text(text):
        """Clean Wikipedia markup from text"""
        if text is None:
            return ""

        # Remove comments
        text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)

        # Remove references
        text = re.sub(r'<ref[^>]*>.*?</ref>', '', text, flags=re.DOTALL)

        # Remove templates
        text = re.sub(r'\{\{[^}]*\}\}', '', text, flags=re.DOTALL)

        # Remove tables
        text = re.sub(r'\{\|[^}]*\|\}', '', text, flags=re.DOTALL)

        # Remove images and files
        text = re.sub(r'\[\[(File|Image):.*?\]\]', '', text, flags=re.DOTALL)

        # Replace links with just the text
        text = re.sub(r'\[\[(.*?)\]\]', lambda m: m.group(1).split('|')[-1], text)

        # Remove formatting
        text = re.sub(r"'{2,}", '', text)

        # Remove section headers
        text = re.sub(r'==+.*?==+', '', text)

        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)

        # Remove multiple spaces and newlines
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    print(f"Processing Wikipedia dump from {wiki_dump_path}")
    articles_processed = 0

    with open(output_file, 'w', encoding='utf-8') as out_file:
        with bz2.BZ2File(wiki_dump_path) as xml_file:
            # Use iterparse to avoid loading entire file into memory
            context = ET.iterparse(xml_file, events=('end',))

            for event, elem in tqdm(context, desc="Processing Wikipedia articles"):
                if elem.tag.endswith('page'):
                    title, text, redirect, namespace = extract_text(elem)

                    # Skip redirects, non-article namespaces, and special pages
                    if (not redirect and namespace == '0' and text and
                            not (title and (title.startswith('Wikipedia:') or
                                            title.startswith('Template:') or
                                            title.startswith('File:') or
                                            title.startswith('Category:')))):

                        # Clean the text
                        clean_content = clean_wiki_text(text)

                        # Skip if too short
                        if len(clean_content.split()) > 50:
                            out_file.write(clean_content + '\n')
                            articles_processed += 1

                            if max_articles and articles_processed >= max_articles:
                                break

                # Clear the element to save memory
                elem.clear()

    print(f"Processed {articles_processed} articles")
    return output_file


# Main function to download Wikipedia and train Word2Vec
def train_word2vec_on_wikipedia(use_cuda=False):
    # Set up directory
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)

    wiki_dump = os.path.join(data_dir, "enwiki-latest-pages-articles.xml.bz2")
    wiki_text = os.path.join(data_dir, "wiki_texts.txt")
    model_file = os.path.join(data_dir, "word2vec_pytorch.txt")

    # Process Wikipedia if needed
    if not os.path.exists(wiki_text):
        if not os.path.exists(wiki_dump):
            print("Please download the Wikipedia dump first from:")
            print("https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2")
            return None

        process_wikipedia(wiki_dump, wiki_text, max_articles=100000)  # Limit for testing

    # Train Word2Vec model
    model = Word2Vec(
        vector_size=300,
        window_size=5,
        min_count=5,
        negative_samples=15,
        learning_rate=0.025,
        epochs=5,
        batch_size=512
    )

    model.train(wiki_text, use_cuda=use_cuda)

    # Save the model
    model.save(model_file)

    return model


def download_text8_corpus(data_dir="data"):
    """Download and extract the text8 corpus (a smaller cleaned Wikipedia subset)"""
    import urllib.request
    import zipfile

    os.makedirs(data_dir, exist_ok=True)
    text8_zip = os.path.join(data_dir, "text8.zip")
    text8_file = os.path.join(data_dir, "text8")

    if not os.path.exists(text8_file):
        if not os.path.exists(text8_zip):
            print("Downloading text8 corpus...")
            url = "http://mattmahoney.net/dc/text8.zip"
            urllib.request.urlretrieve(url, text8_zip)

        print("Extracting text8 corpus...")
        with zipfile.ZipFile(text8_zip, 'r') as zip_ref:
            zip_ref.extractall(data_dir)

    return text8_file


def prepare_text8_for_training(text8_file, output_file):
    """Split text8 into sentences for training"""
    print("Preparing text8 for training...")
    with open(text8_file, 'r', encoding='utf-8') as f:
        text = f.read()

    # Split into chunks of 1000 words to simulate sentences
    words = text.split()
    chunk_size = 1000
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

    with open(output_file, 'w', encoding='utf-8') as f:
        for chunk in chunks:
            f.write(chunk + '\n')

    return output_file


def train_word2vec_on_text8(use_cuda=False):
    """Train Word2Vec on the text8 corpus"""
    # Set up directory
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)

    text8_file = download_text8_corpus(data_dir)
    text8_processed = os.path.join(data_dir, "text8_processed.txt")
    model_file = os.path.join(data_dir, "word2vec_text8.txt")

    # Process text8 if needed
    if not os.path.exists(text8_processed):
        prepare_text8_for_training(text8_file, text8_processed)

    # Create a smaller model for faster training
    model = Word2Vec(
        vector_size=100,  # Smaller embedding size
        window_size=5,
        min_count=5,
        negative_samples=10,
        learning_rate=0.025,
        epochs=3,  # Fewer epochs
        batch_size=512
    )

    # Train on text8
    model.train(text8_processed, use_cuda=use_cuda)

    # Save the model
    model.save(model_file)

    # Test the model with some similar word queries
    test_words = ['king', 'computer', 'man', 'woman', 'book']
    for word in test_words:
        if word in model.word_to_idx:
            similar = model.get_most_similar(word, n=5)
            print(f"\nWords similar to '{word}':")
            for similar_word, similarity in similar:
                print(f"  {similar_word}: {similarity:.4f}")

    return model

# Function to use Word2Vec for Hacker News upvote prediction
def predict_hacker_news_upvotes(model, input_features):
    """
    Create a PyTorch model for Hacker News upvote prediction

    Args:
        model: Trained Word2Vec model
        input_features: Title text or other features of HN posts

    Returns:
        Predicted upvotes
    """

    class HackerNewsPredictor(nn.Module):
        def __init__(self, embedding_dim=300, hidden_dim=128):
            super(HackerNewsPredictor, self).__init__()
            self.layers = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim // 2, 1)
            )

        def forward(self, x):
            return self.layers(x)

    # Convert input text to vectors
    if isinstance(input_features, str):
        # Single title case
        words = model._preprocess_text(input_features)
        vectors = [model.get_vector(word) for word in words if word in model.word_to_idx]

        if vectors:
            # Average word vectors
            doc_vector = np.mean(vectors, axis=0)
        else:
            doc_vector = np.zeros(model.vector_size)

        X = torch.FloatTensor([doc_vector])
    else:
        # Multiple titles case
        X = []
        for title in input_features:
            words = model._preprocess_text(title)
            vectors = [model.get_vector(word) for word in words if word in model.word_to_idx]

            if vectors:
                doc_vector = np.mean(vectors, axis=0)
            else:
                doc_vector = np.zeros(model.vector_size)

            X.append(doc_vector)
        X = torch.FloatTensor(X)

    # Create and train predictor model (this is just a placeholder - would need actual training data)
    predictor = HackerNewsPredictor(embedding_dim=model.vector_size)

    # In a real implementation, you would train this model with actual data
    # For demonstration purposes, we're just returning a random prediction
    with torch.no_grad():
        predictions = predictor(X)

    return predictions.numpy()


# Example usage
if __name__ == "__main__":

    model = train_word2vec_on_text8(use_cuda=True)

    # Train model (this would take a long time with the full Wikipedia dump)
    #model = train_word2vec_on_wikipedia(use_cuda=True)

    # Get embeddings for sample words
    for word in ['neural', 'network', 'computer', 'algorithm']:
        vector = model.get_vector(word)
        print(f"{word}: {vector[:5]}...")  # Print first 5 dimensions

    # Find similar words
    similar_words = model.get_most_similar('computer', n=5)
    print("Words similar to 'computer':")
    for word, similarity in similar_words:
        print(f"  {word}: {similarity:.4f}")

    # Example of HN upvote prediction
    sample_titles = [
        "Show HN: I built a neural network that generates music",
        "Ask HN: What books changed the way you think about programming?",
        "Why I switched from Python to Go for backend development"
    ]

    predictions = predict_hacker_news_upvotes(model, sample_titles)
    for title, pred in zip(sample_titles, predictions):
        print(f"Title: {title}")
        print(f"Predicted upvotes: {int(pred[0])}")
        print()