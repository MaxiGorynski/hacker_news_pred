# Improvements to Word2Vec Implementation

import torch
import torch.nn as nn
import torch.optim as optim
from torch import cosine_similarity
from torch.utils.data import Dataset, DataLoader
import numpy as np
import re
from collections import Counter, defaultdict
import random
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import scipy
import pandas as pd
import numpy as np
import torch
from datetime import datetime



# ----------------------
# 1. Improved Word2Vec Components
# ----------------------

class Word2VecDataset(Dataset):
    def __init__(self, corpus_file, vocab, word_to_idx, discard_probs, window_size=5):
        self.corpus_file = corpus_file
        self.vocab = vocab
        self.word_to_idx = word_to_idx
        self.window_size = window_size
        self.discard_probs = discard_probs

        # Preprocess data and store as list of training pairs
        self.training_pairs = []
        with open(corpus_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Preprocessing corpus"):
                words = self._preprocess_text(line.strip())

                # Apply subsampling
                if self.discard_probs:
                    words = [word for word in words if
                             random.random() > self.discard_probs.get(word, 0)]

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

                    # Add pairs
                    for context_idx in context_indices:
                        self.training_pairs.append((target_idx, context_idx))

        print(f"Total training pairs: {len(self.training_pairs)}")

    def __len__(self):
        return len(self.training_pairs)

    def __getitem__(self, idx):
        """
        Implement __getitem__ method for DataLoader compatibility
        Returns a single training pair (target_word, context_word)
        """
        return self.training_pairs[idx]

    def _preprocess_text(self, text):
        """Basic text preprocessing"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        return text.split()


class EnhancedNegativeSamplingDataset(Dataset):
    def __init__(self, word2vec_dataset, vocab_size, negative_samples=20):
        self.word2vec_dataset = word2vec_dataset
        self.vocab_size = vocab_size
        self.negative_samples = negative_samples

        # Create efficient negative sampling table
        self.negative_table = self._create_negative_table()
        self.training_pairs = self._generate_training_pairs()

    def _create_negative_table(self, table_size=10_000_000):
        """Efficient negative sampling table creation"""
        word_counts = Counter()

        # Sample words to reduce memory usage
        with open(self.word2vec_dataset.corpus_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Building negative sampling table"):
                words = self.word2vec_dataset._preprocess_text(line.strip())
                words = [word for word in words if word in self.word2vec_dataset.word_to_idx]
                word_counts.update(words)

                # Early stopping to prevent excessive memory use
                if len(word_counts) > 750_000:
                    break

        # Compute word frequencies
        word_freqs = np.zeros(self.vocab_size)
        for word, count in word_counts.items():
            if word in self.word2vec_dataset.word_to_idx:
                idx = self.word2vec_dataset.word_to_idx[word]
                word_freqs[idx] = count

        # Apply power law distribution
        word_freqs = np.power(word_freqs, 0.75)
        word_freqs /= np.sum(word_freqs)

        # Create table
        negative_table = []
        for idx, freq in enumerate(word_freqs):
            table_entries = max(1, int(freq * table_size))
            negative_table.extend([idx] * table_entries)

        # Truncate or pad to exact size
        negative_table = negative_table[:table_size] if len(negative_table) > table_size else \
            negative_table + [np.random.randint(0, self.vocab_size)] * (table_size - len(negative_table))

        return np.array(negative_table)

    def _generate_training_pairs(self):
        """Generate training pairs with efficient sampling"""
        training_pairs = []

        with open(self.word2vec_dataset.corpus_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Generating training pairs"):
                words = self.word2vec_dataset._preprocess_text(line.strip())
                indices = [self.word2vec_dataset.word_to_idx.get(word) for word in words]
                indices = [idx for idx in indices if idx is not None]

                for i, target_idx in enumerate(indices):
                    # Dynamic windowing
                    window = random.randint(1, 5)
                    start = max(0, i - window)
                    end = min(len(indices), i + window + 1)

                    context_indices = indices[start:i] + indices[i + 1:end]

                    for context_idx in context_indices:
                        # Simple negative sampling
                        neg_indices = np.random.choice(
                            self.negative_table,
                            size=self.negative_samples
                        )

                        # Ensure no duplicates or overlaps
                        for j, neg_idx in enumerate(neg_indices):
                            if neg_idx == target_idx or neg_idx == context_idx:
                                neg_indices[j] = (neg_idx + 1) % self.vocab_size

                        training_pairs.append((
                            target_idx,
                            context_idx,
                            neg_indices
                        ))

                # Prevent excessive memory usage
                if len(training_pairs) > 10_000_000:
                    break

        return training_pairs

    def __len__(self):
        return len(self.training_pairs)

    def __getitem__(self, idx):
        return self.training_pairs[idx]


class SkipGramNegativeSampling(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()

        # More robust initialization
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # Advanced initialization techniques
        nn.init.xavier_uniform_(self.word_embeddings.weight, gain=1.4)
        nn.init.xavier_uniform_(self.context_embeddings.weight, gain=1.4)

        # Add layer normalization
        self.word_embedding_norm = nn.LayerNorm(embedding_dim)
        self.context_embedding_norm = nn.LayerNorm(embedding_dim)

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
        # Return normalized embeddings for better similarity calculation
        embeddings = self.word_embeddings.weight.data.cpu().numpy()
        norms = np.sqrt(np.sum(embeddings ** 2, axis=1, keepdims=True))
        norms[norms == 0] = 1  # Avoid division by zero
        return embeddings / norms


class Word2Vec:
    def __init__(self, vector_size=50, window_size=5, min_count=3, negative_samples=25,
                 learning_rate=0.0015, epochs=15, batch_size=1026):
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
        self.loss_history = []  # Track loss for plotting

    def _subsample_frequent_words(self, word_counts, total_words, t=1e-5):
        """Subsample frequent words based on formula from original paper"""
        word_probs = {}
        for word, count in word_counts.items():
            # Formula: p(w) = 1 - sqrt(t/f(w))
            freq = count / total_words
            prob = 1.0 - np.sqrt(t / freq)
            word_probs[word] = max(0, prob)
        return word_probs

    def _preprocess_text(self, text):
        """Basic text preprocessing without lemmatization"""
        # Convert to lowercase
        text = text.lower()

        # Remove punctuation and numbers
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        text = re.sub(r'\d+', '', text)  # Remove numbers

        # Split into words
        words = text.split()

        # Optional: Remove very short words
        words = [word for word in words if len(word) > 1]

        return words

    def build_vocabulary(self, corpus_file):
        """Build vocabulary from corpus file"""
        print("Building vocabulary...")
        self._subsample_word_counts = Counter()

        # First pass: count words
        with open(corpus_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(tqdm(f, desc="Counting words")):
                words = self._preprocess_text(line.strip())
                self._subsample_word_counts.update(words)

        total_words = sum(self._subsample_word_counts.values())

        # Filter by minimum count
        self.vocab = [word for word, count in self._subsample_word_counts.items()
                      if count >= self.min_count]

        # Create word-to-index mappings
        for i, word in enumerate(self.vocab):
            self.word_to_idx[word] = i
            self.idx_to_word[i] = word

        print(f"Vocabulary size: {len(self.vocab)}")
        return self._subsample_word_counts

    def train(self, corpus_file, use_cuda=False):
        """Train the Word2Vec model using skip-gram with negative sampling"""
        if not self.vocab:
            self.build_vocabulary(corpus_file)

        # Calculate discard probabilities if not already done
        total_words = sum(self._subsample_word_counts.values())
        discard_probs = self._subsample_frequent_words(
            self._subsample_word_counts,
            total_words
        )

        # Set device
        device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # Create dataset and dataloader
        dataset = Word2VecDataset(
            corpus_file,
            self.vocab,
            self.word_to_idx,
            discard_probs,  # Correctly pass discard probabilities
            window_size=self.window_size
        )
        neg_dataset = EnhancedNegativeSamplingDataset(dataset, len(self.vocab), self.negative_samples)

        # Use DataLoader for more efficient batch processing
        from torch.utils.data import DataLoader
        dataloader = DataLoader(
            neg_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4 if use_cuda else 2,
            pin_memory=use_cuda
        )

        # Create model
        self.model = SkipGramNegativeSampling(len(self.vocab), self.vector_size).to(device)

        # Improved optimizer: Adam instead of SGD
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)

        print("Starting training...")
        for epoch in range(self.epochs):
            total_loss = 0
            batch_count = 0

            # Create progress bar
            progress_bar = tqdm(total=len(dataloader), desc=f"Epoch {epoch + 1}/{self.epochs}")

            # Process batches with DataLoader
            for batch_data in dataloader:
                target_batch, context_batch, neg_batch = batch_data

                # Transfer to device
                targets = target_batch.to(device)
                contexts = context_batch.to(device)
                negatives = neg_batch.to(device)

                # Forward and backward pass
                optimizer.zero_grad()
                loss = self.model(targets, contexts, negatives)
                loss.backward()

                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)

                optimizer.step()

                total_loss += loss.item()
                batch_count += 1

                # Update progress
                progress_bar.update(1)
                progress_bar.set_postfix({"Loss": f"{total_loss / batch_count:.4f}"})

            progress_bar.close()

            # Calculate average loss for this epoch
            avg_loss = total_loss / batch_count
            self.loss_history.append(avg_loss)

            # Update learning rate based on loss
            scheduler.step(avg_loss)

            print(f"Epoch {epoch + 1} completed, Avg. Loss: {avg_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

            # Save checkpoint every few epochs
            if (epoch + 1) % 5 == 0 or epoch == self.epochs - 1:
                self.save(f"data/word2vec_checkpoint_epoch_{epoch + 1}.txt")

        # Plot loss history
        self.plot_loss_history()

        return self.model

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

        # Get word vector (already normalized)
        word_idx = self.word_to_idx[word]
        word_vec = self.model.get_word_embeddings()[word_idx]

        # Compute similarities with all other words
        embeddings = self.model.get_word_embeddings()

        # Compute cosine similarity
        similarities = np.dot(embeddings, word_vec)

        # Get most similar words
        most_similar = []
        indices = np.argsort(-similarities)

        for idx in indices:
            if idx != word_idx:  # Skip the word itself
                most_similar.append((self.idx_to_word[idx], similarities[idx]))
                if len(most_similar) >= n:
                    break

        return most_similar

    def plot_loss_history(self):
        """Plot the loss history during training"""
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.loss_history) + 1), self.loss_history, marker='o')
        plt.title('Training Loss over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig('data/loss_history.png')
        plt.close()


# ----------------------
# 2. Improved Text8 Training
# ----------------------

def train_word2vec_on_text8(use_cuda=False, use_tiny=False, vector_size=300):
    """Train Word2Vec on the text8 corpus with improved parameters"""
    # Set up directory
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)

    # Download and prepare text8
    text8_file = download_text8_corpus(data_dir)

    # Determine which file to use based on use_tiny flag
    if use_tiny:
        text8_processed = os.path.join(data_dir, "text8_tiny.txt")
        model_file = os.path.join(data_dir, "word2vec_text8_tiny_improved.txt")
        create_tiny_text8(text8_file, text8_processed, max_words=500000)
    else:
        text8_processed = os.path.join(data_dir, "text8_processed.txt")
        model_file = os.path.join(data_dir, "word2vec_text8_full.txt")

        # Prepare full corpus if not already processed
        if not os.path.exists(text8_processed):
            prepare_text8_for_training(text8_file, text8_processed)

    # Create model with improved parameters
    model = Word2Vec(
        vector_size=vector_size,  # Use passed vector size
        window_size=8,   # Wider context window
        min_count=5,     # Keep low-frequency words
        negative_samples=15,
        learning_rate=0.0015,
        epochs=15,
        batch_size=2048
    )

    # Train the model
    model.train(text8_processed, use_cuda=use_cuda)

    # Save the model
    model.save(model_file)

    # Test the model with known word pairs
    test_word_pairs = [
        ('king', ['queen', 'prince', 'royal', 'ruler']),
        ('man', ['woman', 'boy', 'person', 'gentleman']),
        ('france', ['paris', 'europe', 'italy', 'spain']),
        ('computer', ['software', 'hardware', 'keyboard', 'programmer']),
        ('good', ['great', 'best', 'better', 'excellent'])
    ]

    print("\nTesting word similarities:")
    for word, expected_similar in test_word_pairs:
        if word in model.word_to_idx:
            similar = model.get_most_similar(word, n=10)
            print(f"\nWords similar to '{word}':")
            for similar_word, similarity in similar:
                print(f"  {similar_word}: {similarity:.4f}")

            # Check if any expected words are in the top 10
            found_expected = [sw for sw, _ in similar if sw in expected_similar]
            if found_expected:
                print(f"  Found expected similar words: {found_expected}")
            else:
                print(f"  No expected similar words found in top results")

    return model


def download_text8_corpus(data_dir="data"):
    """Download and extract the text8 corpus"""
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
    """Split text8 into chunks for training"""
    print("Preparing text8 for training...")
    with open(text8_file, 'r', encoding='utf-8') as f:
        text = f.read()

    # Split into chunks of 1000 words
    words = text.split()
    chunk_size = 1000
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

    with open(output_file, 'w', encoding='utf-8') as f:
        for chunk in chunks:
            f.write(chunk + '\n')

    return output_file


def create_tiny_text8(text8_file, output_file, max_words=500000):
    """Create a smaller text8 corpus for quick testing"""
    print(f"Creating tiny text8 corpus with {max_words} words...")
    with open(text8_file, 'r', encoding='utf-8') as f:
        text = f.read()

    # Take first N words
    words = text.split()[:max_words]

    # Split into chunks
    chunk_size = 1000
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

    with open(output_file, 'w', encoding='utf-8') as f:
        for chunk in chunks:
            f.write(chunk + '\n')

    print(f"Created tiny text8 corpus at {output_file}")
    return output_file


# ----------------------
# 3. Improved Hacker News Predictor
# ----------------------

# Make sure this replaces any existing EnhancedUpvotePredictor class
class EnhancedUpvotePredictor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        # Attention mechanism for input features
        self.attention = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, input_dim),  # Changed to match input_dim
            nn.Softmax(dim=1)
        )

        # Main network with increased width and depth
        self.main_network = nn.Sequential(
            # Layer 1
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),

            # Layer 2
            nn.Linear(512, 384),
            nn.BatchNorm1d(384),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),

            # Layer 3
            nn.Linear(384, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.45),

            # Layer 4
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4),

            # Layer 5
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            # Output layer
            nn.Linear(64, 1),
            nn.ReLU()  # Ensure predictions are non-negative
        )

    def forward(self, x):
        # Apply attention mechanism (fixed version)
        attention_weights = self.attention(x)

        # Element-wise multiplication of input features with attention weights
        attended_input = x * attention_weights

        # Residual connection - combine attended features with original
        enhanced_input = x + attended_input

        # Pass through main network
        return self.main_network(enhanced_input)


def create_enhanced_embedding(title, model, upvotes=None, upvote_mean=None, upvote_median=None, upvote_q75=None):
    """
    Create enhanced document embedding with sophisticated feature engineering

    Args:
        title: The title text
        model: Word2Vec model
        upvotes: Optional upvote count (for training)
        upvote_mean, upvote_median, upvote_q75: Statistics for better feature engineering

    Returns:
        Enhanced document vector with engineered features
    """
    try:
        # Basic preprocessing
        words = model._preprocess_text(str(title))

        if not words:
            # Empty title case
            return np.zeros(model.vector_size + 15)  # Adjust size based on feature count

        # Store word vectors and calculate various statistics
        vectors = []
        word_weights = []
        word_lengths = []

        # Count special words
        hn_markers = ['show', 'ask', 'tell']
        tech_keywords = ['app', 'code', 'software', 'programming', 'data', 'api',
                         'python', 'javascript', 'web', 'neural', 'ai', 'ml']

        # Get IDF-like weights for words (to emphasize rare/important words)
        vocab_size = len(model.vocab)

        for word in words:
            try:
                vec = model.get_vector(word)
                if vec is not None:
                    # Enhanced position weighting (title beginning and end are important)
                    position = words.index(word)
                    if position == 0 or position == len(words) - 1:
                        position_weight = 1.5  # Higher weight for first/last words
                    else:
                        position_weight = 1.0 / (abs(position - len(words) / 2) + 1)

                    # Increase weight for rare words (approximated using index in vocabulary)
                    if word in model.word_to_idx:
                        # Assume lower indices are more common words
                        word_idx = model.word_to_idx[word]
                        rarity_weight = min(3.0, 0.5 + (word_idx / vocab_size) * 2)
                    else:
                        rarity_weight = 1.0

                    # Combine weights
                    combined_weight = position_weight * rarity_weight

                    vectors.append(vec)
                    word_weights.append(combined_weight)
                    word_lengths.append(len(word))
            except Exception:
                continue

        if not vectors:
            # No valid vectors found
            return np.zeros(model.vector_size + 15)  # Adjust size based on feature count

        # Calculate weighted average embedding
        doc_vector = np.average(vectors, axis=0, weights=word_weights)

        # Basic title features
        title_length = len(words)
        title_length_squared = title_length ** 2  # Non-linear length effect
        avg_word_length = np.mean(word_lengths)
        max_word_length = max(word_lengths)
        uppercase_count = sum(1 for word in words if word.isupper())

        # HN-specific features
        has_marker = 1 if any(marker in words for marker in hn_markers) else 0
        marker_position = next((words.index(w) for w in words if w in hn_markers), -1)
        marker_position_normalized = marker_position / len(words) if marker_position >= 0 else -0.1

        # Content type features
        tech_keyword_count = sum(1 for word in words if word in tech_keywords)
        tech_keyword_ratio = tech_keyword_count / len(words) if words else 0

        # Sentiment features (very simple approximation)
        positive_words = ['best', 'great', 'amazing', 'excellent', 'awesome', 'free', 'new']
        negative_words = ['bug', 'issue', 'problem', 'error', 'fail', 'broken']
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        sentiment_score = (positive_count - negative_count) / len(words) if words else 0

        # Combine basic features
        basic_features = [
            title_length,
            title_length_squared,
            avg_word_length,
            max_word_length,
            uppercase_count,
            has_marker,
            marker_position_normalized,
            tech_keyword_count,
            tech_keyword_ratio,
            sentiment_score
        ]

        # Optional: Add upvote-related features for training
        if upvotes is not None and upvote_mean is not None:
            # Log-transform and normalize upvotes
            log_upvotes = np.log1p(upvotes)

            # Create relative indicators
            is_above_mean = 1.0 if upvotes > np.exp(upvote_mean) else 0.0
            is_above_median = 1.0 if upvotes > upvote_median else 0.0
            is_high_performer = 1.0 if upvotes > upvote_q75 else 0.0
            percentile_approx = min(1.0, upvotes / (upvote_q75 * 2))

            upvote_features = [
                log_upvotes,
                is_above_mean,
                is_above_median,
                is_high_performer,
                percentile_approx
            ]

            # Return embedding + all features
            return np.concatenate([doc_vector, basic_features, upvote_features])
        else:
            # For prediction, add placeholder zeros for upvote features
            upvote_features = [0.0, 0.0, 0.0, 0.0, 0.0]
            return np.concatenate([doc_vector, basic_features, upvote_features])

    except Exception as e:
        # Fallback for any errors
        logging.error(f"Error in document embedding: {e}")
        return np.zeros(model.vector_size + 15)  # Adjust size based on feature count

def train_upvote_predictor(model, df, test_size=0.2, validation=True):
    """
        Enhanced training function with better hyperparameters and evaluation

        Args:
            model: Word2Vec model with embeddings
            df: DataFrame with titles and upvotes
            test_size: Fraction of data to use for testing
            validation: Whether to use validation set (if False, use all for training)

        Returns:
            Trained predictor model
        """
    # Get logger
    logger = logging.getLogger(__name__)

    try:
        # Create embeddings
        X = []
        y = []

        logger.info("Creating document embeddings with enhanced features")
        logger.info(f"Processing {len(df)} titles...")

        # Track progress with tqdm
        from tqdm import tqdm

        # Compute statistics for better scaling/normalization
        upvotes_array = np.array(df['upvotes'])
        log_upvotes = np.log1p(upvotes_array)
        upvote_mean = np.mean(log_upvotes)
        upvote_std = np.std(log_upvotes)
        upvote_median = np.median(upvotes_array)
        upvote_q75 = np.percentile(upvotes_array, 75)

        logger.info(f"Upvote statistics - Mean: {upvote_mean:.4f}, Std: {upvote_std:.4f}")
        logger.info(f"Median: {upvote_median}, 75th percentile: {upvote_q75}")

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Creating embeddings"):
            try:
                title = row['title']
                upvotes = row['upvotes']

                # Create enhanced document embedding
                doc_vector = create_enhanced_embedding(
                    title,
                    model,
                    upvotes=upvotes,
                    upvote_mean=upvote_mean,
                    upvote_median=upvote_median,
                    upvote_q75=upvote_q75
                )

                X.append(doc_vector)
                y.append(upvotes)

            except Exception as e:
                logger.error(f"Error processing row {idx}: {e}")
                continue

        logger.info(f"Successfully created {len(X)} document embeddings")

        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)

        logger.info(f"X shape: {X.shape}, y shape: {y.shape}")

        # Apply log transform and normalization to target
        y_log = np.log1p(y)
        y_mean = np.mean(y_log)
        y_std = np.std(y_log)
        y_normalized = (y_log - y_mean) / y_std

        # Split data
        if validation:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_normalized, test_size=test_size, random_state=42
            )
            logger.info(f"Train set: {X_train.shape[0]}, Test set: {X_test.shape[0]}")
        else:
            # Use all data for training (for final model)
            X_train, y_train = X, y_normalized
            X_test, y_test = X[:100], y_normalized[:100]  # Small test set for evaluation
            logger.info(f"Using full dataset for training: {X_train.shape[0]} samples")

        # Convert to tensors
        X_train = torch.FloatTensor(X_train)
        y_train = torch.FloatTensor(y_train).unsqueeze(1)
        X_test = torch.FloatTensor(X_test)
        y_test = torch.FloatTensor(y_test).unsqueeze(1)

        # Create the enhanced predictor
        logger.info(f"Initializing enhanced predictor with input dim: {X_train.shape[1]}")
        predictor = EnhancedUpvotePredictor(X_train.shape[1])

        # Training parameters
        # Lower learning rate and higher weight decay for better regularization
        optimizer = optim.AdamW(
            predictor.parameters(),
            lr=0.0005,  # Lower learning rate for more stable training
            weight_decay=0.03  # Increased weight decay for better regularization
        )

        # Learning rate scheduler for dynamic adjustment
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=7,
            verbose=True
        )

        # Use a combination of MSE (for accuracy) and Huber (for robustness)
        mse_criterion = nn.MSELoss()
        huber_criterion = nn.SmoothL1Loss()

        # Training loop with improved early stopping
        patience = 20  # Increased patience
        best_rmse = float('inf')
        patience_counter = 0
        max_epochs = 150  # Increased maximum epochs

        # Lists for tracking metrics
        train_losses = []
        test_losses = []
        train_rmses = []
        test_rmses = []
        spearman_corrs = []  # Track rank correlation

        # Mini-batch training
        batch_size = 128
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )

        logger.info(f"Starting training with {max_epochs} max epochs, patience={patience}")

        for epoch in range(max_epochs):
            # Training phase
            predictor.train()
            epoch_loss = 0.0
            num_batches = 0

            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()

                # Forward pass
                outputs = predictor(X_batch)

                # Combined loss
                mse_loss = mse_criterion(outputs, y_batch)
                huber_loss = huber_criterion(outputs, y_batch)
                loss = 0.7 * mse_loss + 0.3 * huber_loss  # Weighted combination

                # Backward pass
                loss.backward()

                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(predictor.parameters(), max_norm=1.0)

                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_train_loss = epoch_loss / num_batches
            train_losses.append(avg_train_loss)

            # Evaluation phase
            predictor.eval()
            with torch.no_grad():
                # Test set evaluation
                test_outputs = predictor(X_test)
                test_loss = 0.7 * mse_criterion(test_outputs, y_test) + 0.3 * huber_criterion(test_outputs, y_test)
                test_losses.append(test_loss.item())

                # Denormalize for RMSE calculation
                train_outputs = predictor(X_train)
                train_preds_orig = np.expm1(train_outputs.numpy() * y_std + y_mean)
                train_true_orig = np.expm1(y_train.numpy() * y_std + y_mean)
                test_preds_orig = np.expm1(test_outputs.numpy() * y_std + y_mean)
                test_true_orig = np.expm1(y_test.numpy() * y_std + y_mean)

                # Calculate RMSE
                train_rmse = np.sqrt(np.mean((train_preds_orig - train_true_orig) ** 2))
                test_rmse = np.sqrt(np.mean((test_preds_orig - test_true_orig) ** 2))

                train_rmses.append(train_rmse)
                test_rmses.append(test_rmse)

                # Calculate Spearman's rank correlation
                from scipy.stats import spearmanr
                corr, _ = spearmanr(test_preds_orig.flatten(), test_true_orig.flatten())
                spearman_corrs.append(corr)

                # Log progress
                if epoch % 5 == 0:
                    logger.info(f"Epoch {epoch + 1}/{max_epochs}")
                    logger.info(f"Train Loss: {avg_train_loss:.6f}, Test Loss: {test_loss.item():.6f}")
                    logger.info(f"Train RMSE: {train_rmse:.2f}, Test RMSE: {test_rmse:.2f}")
                    logger.info(f"Spearman Correlation: {corr:.4f}")
                    logger.info(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")

                # Update learning rate based on test loss
                scheduler.step(test_loss.item())

                # Check for improvement
                if test_rmse < best_rmse:
                    improvement = (best_rmse - test_rmse) / best_rmse * 100
                    best_rmse = test_rmse
                    patience_counter = 0

                    # Save the best model
                    torch.save(predictor.state_dict(), 'best_predictor_model.pth')
                    logger.info(f"New best model saved with RMSE: {best_rmse:.2f} (improved by {improvement:.2f}%)")
                else:
                    patience_counter += 1

                # Early stopping check
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

        # Load the best model for return
        try:
            predictor.load_state_dict(torch.load('best_predictor_model.pth'))
            logger.info("Loaded best model for return")

            # Plot training metrics
            try:
                import matplotlib.pyplot as plt

                # Create figure with 3 subplots
                plt.figure(figsize=(18, 6))

                # Plot loss
                plt.subplot(1, 3, 1)
                plt.plot(train_losses, label='Train Loss')
                plt.plot(test_losses, label='Test Loss')
                plt.title('Loss During Training')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend()

                # Plot RMSE
                plt.subplot(1, 3, 2)
                plt.plot(train_rmses, label='Train RMSE')
                plt.plot(test_rmses, label='Test RMSE')
                plt.title('RMSE During Training')
                plt.xlabel('Epoch')
                plt.ylabel('RMSE')
                plt.legend()

                # Plot Spearman correlation
                plt.subplot(1, 3, 3)
                plt.plot(spearman_corrs)
                plt.title('Spearman Rank Correlation')
                plt.xlabel('Epoch')
                plt.ylabel('Correlation')
                plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)

                plt.tight_layout()
                plt.savefig('upvote_predictor_metrics.png')
                logger.info("Saved training metrics plot")
            except Exception as plot_error:
                logger.error(f"Error creating plots: {plot_error}")
        except Exception as load_error:
            logger.error(f"Error loading best model: {load_error}")

        # Final evaluation
        logger.info(f"Final best RMSE: {best_rmse:.2f}")
        logger.info(f"Final Spearman Correlation: {spearman_corrs[-1]:.4f}")

        # Save y normalization parameters for inference
        try:
            normalization_params = {
                'y_mean': float(y_mean),
                'y_std': float(y_std)
            }
            with open('normalization_params.json', 'w') as f:
                import json
                json.dump(normalization_params, f)
            logger.info("Saved normalization parameters for inference")
        except Exception as e:
            logger.error(f"Error saving normalization parameters: {e}")

        return predictor, y_mean, y_std

    except Exception as e:
        logger.error(f"Unexpected error in upvote predictor training: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None, None, None


def generate_synthetic_hn_data(base_data, additional_samples=1000):
    synthetic_data = []

    # More sophisticated generation
    contexts = [
        "Web Development", "Machine Learning", "Startup Tech",
        "Open Source", "Cloud Computing", "AI/ML", "Cybersecurity"
    ]

    for _ in range(additional_samples):
        context = random.choice(contexts)
        keywords = {
            "Web Development": ["React", "Vue", "Angular", "Frontend"],
            "Machine Learning": ["TensorFlow", "PyTorch", "ML", "Deep Learning"],
            "Startup Tech": ["Funding", "Pitch", "Innovation"],
            "Open Source": ["GitHub", "Community", "Collaboration"],
            "Cloud Computing": ["AWS", "Azure", "Kubernetes"],
            "AI/ML": ["GPT", "Neural Networks", "Machine Learning"],
            "Cybersecurity": ["Encryption", "Penetration Testing", "Security"]
        }

        # Generate more realistic title
        title_template = random.choice([
            "Show HN: {tech} {context} Project",
            "Ask HN: {context} Best Practices",
            "Tell HN: My Journey with {tech}",
            "How I Built a {tech} Solution for {context}"
        ])

        tech = random.choice(keywords.get(context, []))
        title = title_template.format(tech=tech, context=context)

        # More nuanced upvote generation
        base_upvotes = np.random.lognormal(mean=5, sigma=1) * 50
        context_boost = {
            "Web Development": 1.2,
            "Machine Learning": 1.5,
            "Startup Tech": 1.3,
            "Open Source": 1.1,
            "Cloud Computing": 1.4,
            "AI/ML": 1.6,
            "Cybersecurity": 1.2
        }

        upvotes = int(base_upvotes * context_boost.get(context, 1.0))

        synthetic_data.append((title, upvotes))

    return base_data + synthetic_data


def create_document_embedding(title, model, additional_features=True):
    """
    More sophisticated document embedding with:
    - TF-IDF weighting
    - Semantic coherence scoring
    - Advanced feature engineering
    """
    words = model._preprocess_text(title)

    # More sophisticated embedding
    vectors = []
    word_weights = []

    for word in words:
        vec = model.get_vector(word)
        if vec is not None:
            # Weight words based on position and uniqueness
            position_weight = 1.0 / (abs(words.index(word) - len(words) / 2) + 1)
            unique_word_weight = 1.0 / (words.count(word) ** 0.5)

            vectors.append(vec)
            word_weights.append(position_weight * unique_word_weight)

    if vectors:
        # Weighted average embedding
        doc_vector = np.average(vectors, axis=0, weights=word_weights)

        # Add additional engineered features
        features = [
            len(words),  # Title length
            sum(1 for word in words if word.isupper()),  # Uppercase word count
            sum(1 for word in words if word.startswith('HN:')),  # HN-specific markers
            np.mean([len(word) for word in words]),  # Average word length
            sum(1 for word in ['show', 'ask', 'tell'] if word in words)  # HN post type
        ]

        return np.concatenate([doc_vector, features])

    return np.zeros(model.vector_size + 5)


def cache_embeddings(X, y, logger=None):
    """
    Cache the document embeddings and labels to disk for faster reloading.

    Parameters:
    -----------
    X : numpy.ndarray
        Document embeddings with shape (n_samples, features)
    y : numpy.ndarray
        Upvote counts with shape (n_samples,)
    logger : logging.Logger, optional
        Logger for status messages
    """
    import numpy as np
    import pickle
    import os

    if logger is None:
        import logging
        logger = logging.getLogger(__name__)

    # Save as pickle for faster loading
    try:
        logger.info("Caching document embeddings to disk...")
        with open('document_embeddings.pkl', 'wb') as f:
            pickle.dump({'X': X, 'y': y}, f)
        logger.info(f"Saved document embeddings to document_embeddings.pkl")
    except Exception as e:
        logger.warning(f"Could not save embeddings as pickle: {e}")

        # Fallback to numpy arrays
        try:
            np.save('X_embeddings.npy', X)
            np.save('y_values.npy', y)
            logger.info("Saved document embeddings as numpy arrays")
        except Exception as e2:
            logger.error(f"Failed to save embeddings: {e2}")
            return False

    return True

# ----------------------
# 4. Hacker News Corpus
# ----------------------

def load_hacker_news_corpus(csv_path, limit=None):
    """
    Load Hacker News titles from CSV

    Args:
    - csv_path: Path to the CSV file
    - limit: Optional limit on number of rows to process

    Returns:
    - Processed text file for Word2Vec training
    - Original dataframe
    """
    # Read CSV
    df = pd.read_csv(csv_path)

    # Optional: limit rows
    if limit:
        df = df.head(limit)

    # Preprocess titles
    df['processed_title'] = df['title'].apply(lambda x: ' '.join(
        word.lower()
        for word in str(x).split()
        if len(word) > 1  # Remove very short words
    ))

    # Create text corpus file
    output_file = 'processed_hn_corpus.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        for title in df['processed_title']:
            f.write(title + '\n')

    return output_file, df


def create_upvote_enhanced_embedding(title, model, upvotes=None):
    """
    Enhanced document embedding that incorporates upvote information with improved error handling
    """
    # Get logger
    logger = logging.getLogger(__name__)

    try:
        words = model._preprocess_text(str(title))  # Ensure title is a string

        vectors = []
        word_weights = []

        for word in words:
            try:
                vec = model.get_vector(word)
                if vec is not None:
                    # Position-based weighting
                    position_weight = 1.0 / (abs(words.index(word) - len(words) / 2) + 1)
                    unique_word_weight = 1.0 / (words.count(word) ** 0.5)

                    vectors.append(vec)
                    word_weights.append(position_weight * unique_word_weight)
            except Exception as word_error:
                # Just skip words that can't be processed
                continue

        if vectors:
            # Weighted average embedding
            doc_vector = np.average(vectors, axis=0, weights=word_weights)

            # Additional features
            features = [
                len(words),  # Title length
                sum(1 for word in words if word.isupper()),  # Uppercase word count
                sum(1 for word in ['show', 'ask', 'tell'] if word in words),  # HN post type
                np.mean([len(word) for word in words]) if words else 0  # Average word length
            ]

            # Optional: Incorporate upvote-related features
            if upvotes is not None:
                try:
                    # Log-transform upvotes to reduce skew
                    upvote_features = [
                        np.log1p(upvotes),  # Log-transformed upvotes
                        1 if upvotes > 0 else 0  # Binary indicator for any upvotes
                    ]
                    features.extend(upvote_features)
                except Exception as upvote_error:
                    # If there's an issue with upvotes, add zeros
                    features.extend([0, 0])

            return np.concatenate([doc_vector, features])
        else:
            # No valid vectors, return zeros
            return np.zeros(model.vector_size + (6 if upvotes is not None else 4))
    except Exception as e:
        # Fallback for any errors - return zeros
        return np.zeros(model.vector_size + (6 if upvotes is not None else 4))


def train_hacker_news_upvote_predictor(model, df):
    """
    Train upvote predictor using Hacker News dataset with enhanced error handling and logging
    """
    # Get logger
    logger = logging.getLogger(__name__)

    try:
        # Create embeddings
        X = []
        y = []

        logger.info("Creating document embeddings")
        logger.info(f"Processing {len(df)} titles for embedding creation")

        # Track progress with tqdm
        from tqdm import tqdm
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Creating embeddings"):
            try:
                title = row['title']
                upvotes = row['upvotes']

                # Log periodic updates
                if idx % 1000 == 0:
                    logger.info(f"Processing embedding {idx}/{len(df)}")

                doc_vector = create_upvote_enhanced_embedding(
                    title,
                    model,
                    upvotes
                )
                X.append(doc_vector)
                y.append(upvotes)

            except KeyError as ke:
                logger.error(f"Missing required column in dataframe at index {idx}: {ke}")
                continue
            except Exception as e:
                logger.error(f"Error processing row {idx}: {e}")
                continue

        logger.info(f"Successfully created {len(X)} document embeddings")

        if not X:
            logger.error("No valid embeddings created. Cannot train model.")
            return None

        try:
            X = np.array(X)
            y = np.array(y)

            logger.info(f"X shape: {X.shape}, y shape: {y.shape}")

            # Log transform target variable
            y_log = np.log1p(y)
            y_normalized = (y_log - np.mean(y_log)) / np.std(y_log)

            logger.info("Splitting data into train/test sets")
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_normalized, test_size=0.2, random_state=42
            )

            logger.info(f"Train set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")

            # Convert to tensors
            X_train = torch.FloatTensor(X_train)
            y_train = torch.FloatTensor(y_train).unsqueeze(1)
            X_test = torch.FloatTensor(X_test)
            y_test = torch.FloatTensor(y_test).unsqueeze(1)

            logger.info("Initializing predictor model")
            # Use existing EnhancedUpvotePredictor
            predictor = EnhancedUpvotePredictor(X_train.shape[1])

            # Log model summary
            logger.info(f"Predictor input dimension: {X_train.shape[1]}")
            logger.info(f"Model architecture: {predictor}")

            # Training setup
            optimizer = optim.AdamW(
                predictor.parameters(),  # This should now work with proper nn.Module inheritance
                lr=0.001,
                weight_decay=0.01
            )

            criterion = nn.SmoothL1Loss()

            # Training loop with early stopping
            best_rmse = float('inf')
            patience = 10
            patience_counter = 0

            # Lists to track training progress
            train_losses = []
            test_losses = []
            train_rmses = []
            test_rmses = []

            logger.info("Starting Upvote Predictor Training Loop")

            # Use tqdm for progress tracking
            max_epochs = 50
            for epoch in tqdm(range(max_epochs), desc="Training epochs"):
                # Training phase
                predictor.train()
                optimizer.zero_grad()

                # Forward pass
                outputs = predictor(X_train)
                loss = criterion(outputs, y_train)

                # Backward pass
                loss.backward()
                optimizer.step()

                # Evaluation phase
                predictor.eval()
                with torch.no_grad():
                    # Train set evaluation
                    train_outputs = predictor(X_train)
                    train_loss = criterion(train_outputs, y_train)

                    # Test set evaluation
                    test_outputs = predictor(X_test)
                    test_loss = criterion(test_outputs, y_test)

                    # Denormalize predictions
                    train_preds_orig = np.expm1(train_outputs.numpy() * np.std(y_log) + np.mean(y_log))
                    train_true_orig = np.expm1(y_train.numpy() * np.std(y_log) + np.mean(y_log))
                    test_preds_orig = np.expm1(test_outputs.numpy() * np.std(y_log) + np.mean(y_log))
                    test_true_orig = np.expm1(y_test.numpy() * np.std(y_log) + np.mean(y_log))

                    # Calculate RMSE
                    train_rmse = np.sqrt(np.mean((train_preds_orig - train_true_orig) ** 2))
                    test_rmse = np.sqrt(np.mean((test_preds_orig - test_true_orig) ** 2))

                    # Log training progress
                    logger.info(f"Epoch {epoch + 1}/{max_epochs}")
                    logger.info(f"Train Loss: {train_loss.item():.4f}, Train RMSE: {train_rmse:.2f}")
                    logger.info(f"Test Loss: {test_loss.item():.4f}, Test RMSE: {test_rmse:.2f}")

                    # Ensure logs are flushed
                    sys.stdout.flush()
                    sys.stderr.flush()

                    # Track metrics
                    train_losses.append(train_loss.item())
                    test_losses.append(test_loss.item())
                    train_rmses.append(train_rmse)
                    test_rmses.append(test_rmse)

                    # Early stopping
                    if test_rmse < best_rmse:
                        best_rmse = test_rmse
                        patience_counter = 0
                        # Save the best model
                        torch.save(predictor.state_dict(), 'best_predictor_model.pth')
                        logger.info(f"New best model saved with RMSE: {best_rmse:.2f}")
                    else:
                        patience_counter += 1

                    # Stop if no improvement
                    if patience_counter >= patience:
                        logger.info(f"Early stopping at epoch {epoch + 1}")
                        break

            # Final logging
            logger.info(f"Training completed. Best Test RMSE: {best_rmse:.2f}")

            # Plot training progress
            try:
                import matplotlib.pyplot as plt

                # Plot losses
                plt.figure(figsize=(12, 5))
                plt.subplot(1, 2, 1)
                plt.plot(train_losses, label='Train Loss')
                plt.plot(test_losses, label='Test Loss')
                plt.title('Loss During Training')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend()

                # Plot RMSE
                plt.subplot(1, 2, 2)
                plt.plot(train_rmses, label='Train RMSE')
                plt.plot(test_rmses, label='Test RMSE')
                plt.title('RMSE During Training')
                plt.xlabel('Epoch')
                plt.ylabel('RMSE')
                plt.legend()

                plt.tight_layout()
                plt.savefig('upvote_predictor_training.png')
                logger.info("Training progress plots saved to 'upvote_predictor_training.png'")
            except Exception as plot_error:
                logger.warning(f"Could not create training plots: {plot_error}")

            return predictor

        except Exception as training_error:
            logger.error(f"Error during model training: {training_error}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    except Exception as e:
        logger.error(f"Unexpected error in upvote predictor training: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


# ----------------------
# 5. Sample Usage
# ----------------------

import logging
import sys
import os


def setup_logging():
    """Set up comprehensive logging with redirected stdout/stderr"""
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s: %(message)s',
        handlers=[
            # Log to console
            logging.StreamHandler(sys.stdout),
            # Log to file
            logging.FileHandler('logs/model_training.log', mode='w')
        ]
    )

    # Redirect stdout and stderr to logging
    class LoggerWriter:
        def __init__(self, logger, level):
            self.logger = logger
            self.level = level
            self.buffer = []

        def write(self, message):
            if message.strip():
                self.logger.log(self.level, message.rstrip())

        def flush(self):
            pass

    # Redirect stdout and stderr to logging
    sys.stdout = LoggerWriter(logging.getLogger('STDOUT'), logging.INFO)
    sys.stderr = LoggerWriter(logging.getLogger('STDERR'), logging.ERROR)

    return logging.getLogger(__name__)


def main():
    # Set up logging
    logger = setup_logging()

    try:
        # Log start of process
        logger.info("Starting Enhanced Hacker News Upvote Predictor Training")
        logger.info(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Load and preprocess Hacker News corpus - use more data (100K instead of 50K)
        logger.info("Loading Hacker News corpus")
        try:
            corpus_file, df = load_hacker_news_corpus('df_200K.csv', limit=100000)
            logger.info(f"Corpus loaded. Total titles: {len(df)}")

            # Check DataFrame structure and analyze data distribution
            logger.info(f"DataFrame columns: {df.columns.tolist()}")
            logger.info(f"Upvotes range: {df['upvotes'].min()} to {df['upvotes'].max()}")

            # Calculate and log statistics about upvotes
            upvotes = df['upvotes'].values
            logger.info(f"Upvote statistics:")
            logger.info(f"  Mean: {np.mean(upvotes):.2f}")
            logger.info(f"  Median: {np.median(upvotes):.2f}")
            logger.info(f"  Std Dev: {np.std(upvotes):.2f}")
            logger.info(f"  25th percentile: {np.percentile(upvotes, 25):.2f}")
            logger.info(f"  75th percentile: {np.percentile(upvotes, 75):.2f}")
            logger.info(f"  90th percentile: {np.percentile(upvotes, 90):.2f}")

            # Optional: Plot histogram of upvotes (helps understand distribution)
            try:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(10, 6))
                plt.hist(np.clip(upvotes, 0, 500), bins=50)
                plt.title('Distribution of Upvotes (clipped at 500)')
                plt.xlabel('Upvotes')
                plt.ylabel('Frequency')
                plt.savefig('upvote_distribution.png')
                logger.info("Saved upvote distribution histogram")
            except Exception as plot_error:
                logger.warning(f"Error creating histogram: {plot_error}")

        except Exception as corpus_error:
            logger.error(f"Error loading corpus: {corpus_error}")
            import traceback
            logger.error(traceback.format_exc())
            raise

        # Ensure logs are flushed
        sys.stdout.flush()
        sys.stderr.flush()

        # Load pre-trained Word2Vec model
        logger.info("Loading pre-trained Word2Vec Model")
        try:
            # Initialize model
            model = Word2Vec(
                vector_size=100,  # Make sure this matches your original model
                window_size=5,
                min_count=3,
                negative_samples=25,
                learning_rate=0.0015,
                epochs=15,
                batch_size=1024
            )

            # Load existing weights
            word2vec_weights_path = 'HN_Corpus_Model_Weights.txt'
            model.load(word2vec_weights_path)
            logger.info(f"Successfully loaded Word2Vec weights with {len(model.vocab)} words")

            # Verify a few key words exist in the vocabulary
            important_words = ['show', 'ask', 'tell', 'how', 'why', 'what', 'guide', 'tutorial']
            found_words = [word for word in important_words if word in model.word_to_idx]
            logger.info(f"Found {len(found_words)}/{len(important_words)} important keywords")

        except Exception as load_error:
            logger.error(f"Error loading Word2Vec weights: {load_error}")
            import traceback
            logger.error(traceback.format_exc())
            raise

        # Train enhanced upvote predictor
        logger.info("Starting Enhanced Upvote Predictor Training")
        try:
            predictor, y_mean, y_std = train_hacker_news_upvote_predictor(model, df)

            if predictor is not None:
                logger.info("Enhanced Upvote Predictor Training Completed Successfully")
            else:
                logger.error("Upvote Predictor Training Failed")
                return model, None

        except Exception as predictor_error:
            logger.error(f"Error during Enhanced Upvote Predictor training: {predictor_error}")
            import traceback
            logger.error(traceback.format_exc())
            return model, None

        # Save the trained predictor
        if predictor is not None:
            try:
                # Save full model in PyTorch format
                enhanced_predictor_path = 'Enhanced_HN_Predictor.pth'
                torch.save(predictor.state_dict(), enhanced_predictor_path)
                logger.info(f"Saved enhanced predictor to {enhanced_predictor_path}")

                # Also save a readable summary
                readable_path = 'Enhanced_HN_Predictor_Summary.txt'
                with open(readable_path, 'w') as f:
                    # Model architecture
                    f.write(f"Enhanced Upvote Predictor Architecture:\n")
                    f.write(f"{predictor}\n\n")

                    # Input dimension
                    input_dim = predictor.main_network[0].in_features
                    f.write(f"Input dimension: {input_dim}\n\n")

                    # Normalization parameters
                    f.write(f"Normalization parameters:\n")
                    f.write(f"  y_mean: {y_mean}\n")
                    f.write(f"  y_std: {y_std}\n\n")

                    # Parameter counts
                    total_params = sum(p.numel() for p in predictor.parameters())
                    trainable_params = sum(p.numel() for p in predictor.parameters() if p.requires_grad)
                    f.write(f"Total parameters: {total_params:,}\n")
                    f.write(f"Trainable parameters: {trainable_params:,}\n\n")

                    # Sample parameters (first few values of each layer)
                    f.write("Sample parameter values:\n")
                    for name, param in predictor.named_parameters():
                        if param.dim() > 0:  # Skip empty parameters
                            f.write(f"  {name}: Shape {param.shape}\n")
                            flat_values = param.detach().numpy().flatten()[:5]
                            f.write(f"    First 5 values: {flat_values}\n")

                logger.info(f"Saved readable model summary to {readable_path}")

                # Save normalization parameters for inference
                norm_params_path = 'normalization_params.json'
                with open(norm_params_path, 'w') as f:
                    import json
                    json.dump({
                        'y_mean': float(y_mean),
                        'y_std': float(y_std)
                    }, f)
                logger.info(f"Saved normalization parameters to {norm_params_path}")

                # Create a sample test script for interactive evaluation
                test_script_path = 'test_enhanced_predictor.py'
                with open(test_script_path, 'w') as f:
                    f.write("""
import torch
import torch.nn as nn
import numpy as np
import json
import sys
import os
import re

# Define the necessary model classes
class EnhancedUpvotePredictor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        # Attention mechanism for input features
        self.attention = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Softmax(dim=1)
        )

        # Main network with increased width and depth
        self.main_network = nn.Sequential(
            # Layer 1
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),

            # Layer 2
            nn.Linear(512, 384),
            nn.BatchNorm1d(384),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),

            # Layer 3
            nn.Linear(384, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.45),

            # Layer 4
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4),

            # Layer 5
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            # Output layer
            nn.Linear(64, 1),
            nn.ReLU()  # Ensure predictions are non-negative
        )

    def forward(self, x):
        # Apply attention mechanism
        batch_size = x.shape[0]
        attention_weights = self.attention(x.unsqueeze(2).transpose(1, 2))
        attended_input = torch.bmm(attention_weights.transpose(1, 2), x.unsqueeze(2)).squeeze(2)

        # Residual connection - combine attended features with original
        enhanced_input = x + attended_input

        # Pass through main network
        return self.main_network(enhanced_input)

class Word2Vec:
    def __init__(self, vector_size=100):
        self.vector_size = vector_size
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.vocab = []
        self.embeddings = None

    def _preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'[^\\w\\s]', '', text)
        text = re.sub(r'\\d+', '', text)
        return [word for word in text.split() if len(word) > 1]

    def load(self, filename):
        with open(filename, 'r', encoding='utf-8') as f:
            header = f.readline().split()
            vocab_size = int(header[0])
            self.vector_size = int(header[1])

            self.embeddings = np.zeros((vocab_size, self.vector_size))

            for i, line in enumerate(f):
                parts = line.rstrip().split(' ')
                word = parts[0]

                self.word_to_idx[word] = i
                self.idx_to_word[i] = word
                self.vocab.append(word)

                self.embeddings[i] = np.array([float(x) for x in parts[1:]])

        return self

    def get_vector(self, word):
        if word in self.word_to_idx:
            idx = self.word_to_idx[word]
            return self.embeddings[idx]
        return None

def create_enhanced_embedding(title, model):
    \"\"\"Simplified embedding creation for prediction\"\"\"
    words = model._preprocess_text(str(title))

    if not words:
        return np.zeros(model.vector_size + 15)

    vectors = []
    word_weights = []
    word_lengths = []

    hn_markers = ['show', 'ask', 'tell']
    tech_keywords = ['app', 'code', 'software', 'programming', 'data', 'api', 
                     'python', 'javascript', 'web', 'neural', 'ai', 'ml']

    for word in words:
        vec = model.get_vector(word)
        if vec is not None:
            position = words.index(word)
            position_weight = 1.5 if position == 0 or position == len(words) - 1 else 1.0
            vectors.append(vec)
            word_weights.append(position_weight)
            word_lengths.append(len(word))

    if not vectors:
        return np.zeros(model.vector_size + 15)

    doc_vector = np.average(vectors, axis=0, weights=word_weights)

    title_length = len(words)
    title_length_squared = title_length ** 2
    avg_word_length = np.mean(word_lengths)
    max_word_length = max(word_lengths)
    uppercase_count = sum(1 for word in words if word.isupper())

    has_marker = 1 if any(marker in words for marker in hn_markers) else 0
    marker_position = next((words.index(w) for w in words if w in hn_markers), -1)
    marker_position_normalized = marker_position / len(words) if marker_position >= 0 else -0.1

    tech_keyword_count = sum(1 for word in words if word in tech_keywords)
    tech_keyword_ratio = tech_keyword_count / len(words)

    positive_words = ['best', 'great', 'amazing', 'excellent', 'awesome', 'free', 'new']
    negative_words = ['bug', 'issue', 'problem', 'error', 'fail', 'broken']
    positive_count = sum(1 for word in words if word in positive_words)
    negative_count = sum(1 for word in words if word in negative_words)
    sentiment_score = (positive_count - negative_count) / len(words)

    features = [
        title_length,
        title_length_squared,
        avg_word_length,
        max_word_length,
        uppercase_count,
        has_marker,
        marker_position_normalized,
        tech_keyword_count,
        tech_keyword_ratio,
        sentiment_score,
        0.0, 0.0, 0.0, 0.0, 0.0  # Placeholder values for upvote features
    ]

    return np.concatenate([doc_vector, features])

def predict_upvotes(title):
    # Load normalization parameters
    with open('normalization_params.json', 'r') as f:
        norm_params = json.load(f)
    y_mean = norm_params['y_mean']
    y_std = norm_params['y_std']

    # Load Word2Vec model
    model = Word2Vec()
    model.load('HN_Corpus_Model_Weights.txt')

    # Create document embedding
    doc_vector = create_enhanced_embedding(title, model)

    # Convert to tensor
    X = torch.FloatTensor(doc_vector).unsqueeze(0)

    # Load model
    predictor = EnhancedUpvotePredictor(X.shape[1])
    predictor.load_state_dict(torch.load('Enhanced_HN_Predictor.pth'))
    predictor.eval()

    # Make prediction
    with torch.no_grad():
        normalized_prediction = predictor(X).item()

    # Denormalize
    prediction = np.expm1(normalized_prediction * y_std + y_mean)
    return max(0, int(round(prediction)))

def main():
    if len(sys.argv) > 1:
        # Use command line argument as title
        title = ' '.join(sys.argv[1:])
    else:
        # Interactive mode
        title = input("Enter a Hacker News title: ")

    score = predict_upvotes(title)
    print(f"Predicted score: {score}")

if __name__ == "__main__":
    main()
""")

                logger.info(f"Created sample test script at {test_script_path}")

            except Exception as save_error:
                logger.error(f"Error saving enhanced predictor: {save_error}")
                import traceback
                logger.error(traceback.format_exc())

        logger.info("Enhanced Model Training and Saving Process Completed Successfully")
        return model, predictor, y_mean, y_std

    except Exception as e:
        logger.error(f"Unexpected error during training: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()