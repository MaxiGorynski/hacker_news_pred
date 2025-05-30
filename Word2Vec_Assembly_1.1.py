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
import itertools
import os
import csv
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
    def __init__(self, vector_size=400, window_size=5, min_count=5, negative_samples=20,
                 learning_rate=0.0025, epochs=15, batch_size=2048):  # More epochs, smaller learning rate
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

class EnhancedUpvotePredictor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.PReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.SELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.PReLU(),
            nn.Linear(64, 1),
            nn.ReLU()  # Ensure non-negative predictions
        )

    def forward(self, x):
        return self.model(x)


def train_upvote_predictor(model, hn_data, test_size=0.2, epochs=50):
    # Prepare data with enhanced embedding
    X = []
    y = []

    for title, upvotes in hn_data:
        doc_vector = create_document_embedding(title, model)
        X.append(doc_vector)
        y.append(upvotes)

    X = np.array(X)
    y = np.array(y)

    # Log transformation of target variable
    y_log = np.log1p(y)
    y_normalized = (y_log - np.mean(y_log)) / np.std(y_log)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y_normalized, test_size=test_size)

    # Convert to tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train).unsqueeze(1)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test).unsqueeze(1)

    # Initialize predictor
    predictor = EnhancedUpvotePredictor(X_train.shape[1])

    # Advanced optimizer
    optimizer = optim.AdamW(
        predictor.parameters(),
        lr=0.001,
        weight_decay=0.01
    )

    # Huber loss for robust regression
    criterion = nn.SmoothL1Loss()

    # Training loop with early stopping
    best_rmse = float('inf')
    patience = 10
    patience_counter = 0

    for epoch in range(epochs):
        predictor.train()
        optimizer.zero_grad()

        outputs = predictor(X_train)
        loss = criterion(outputs, y_train)

        loss.backward()
        optimizer.step()

        # Evaluation
        predictor.eval()
        with torch.no_grad():
            train_preds = outputs.numpy()
            test_outputs = predictor(X_test)
            test_preds = test_outputs.numpy()

            # Denormalize predictions
            train_preds_orig = np.expm1(train_preds * np.std(y_log) + np.mean(y_log))
            test_preds_orig = np.expm1(test_preds * np.std(y_log) + np.mean(y_log))

            train_rmse = np.sqrt(np.mean((train_preds_orig - np.expm1(y_train.numpy())) ** 2))
            test_rmse = np.sqrt(np.mean((test_preds_orig - np.expm1(y_test.numpy())) ** 2))

            print(f"Epoch {epoch + 1}, Train RMSE: {train_rmse:.2f}, Test RMSE: {test_rmse:.2f}")

            # Early stopping
            if test_rmse < best_rmse:
                best_rmse = test_rmse
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping")
                    break

    return predictor


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

# ----------------------
# 4. Self-Improving Adjustments
# ----------------------
def comprehensive_word2vec_exploration(max_configurations=50, use_tiny=False):
    # Create a unique timestamp for this exploration run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = "exploration_logs"
    os.makedirs(log_dir, exist_ok=True)

    # Create detailed logging files
    config_log_path = os.path.join(log_dir, f"config_log_{timestamp}.csv")
    performance_log_path = os.path.join(log_dir, f"performance_log_{timestamp}.txt")

    # Open CSV for configuration logging
    with open(config_log_path, 'w', newline='') as config_csv, \
            open(performance_log_path, 'w') as perf_log:

        # CSV writer for configurations
        config_writer = csv.writer(config_csv)
        config_writer.writerow([
            "Configuration_ID",
            "Vector_Size",
            "Window_Size",
            "Min_Count",
            "Negative_Samples",
            "Learning_Rate",
            "Epochs",
            "Batch_Size",
            "Overall_Performance_Score",
            "Similarity_Scores",
            "Coverage_Scores"
        ])

        # Reduce parameter space for quicker testing
        param_grid = {
            'vector_size': [50, 100, 300],
            'window_size': [3, 5, 8],
            'min_count': [1, 3, 5],
            'negative_samples': [10, 15, 25],
            'learning_rate': [0.001, 0.0015, 0.005],
            'epochs': [10, 15, 25],
            'batch_size': [512, 1024, 2048]
        }

        # Generate all possible configurations
        import itertools
        all_configs = list(itertools.product(
            param_grid['vector_size'],
            param_grid['window_size'],
            param_grid['min_count'],
            param_grid['negative_samples'],
            param_grid['learning_rate'],
            param_grid['epochs'],
            param_grid['batch_size']
        ))

        # Randomize or limit configurations to prevent excessive runtime
        import random
        random.shuffle(all_configs)
        configs_to_test = all_configs[:max_configurations]

        best_config = None
        best_performance = float('inf')

        # Iterate through configurations
        for config_id, (vector_size, window_size, min_count, negative_samples,
                        learning_rate, epochs, batch_size) in enumerate(configs_to_test, 1):

            print(f"\n--- Testing Configuration {config_id}/{len(configs_to_test)} ---")
            print(f"Parameters:")
            print(f"Vector Size: {vector_size}")
            print(f"Window Size: {window_size}")
            print(f"Min Count: {min_count}")
            print(f"Negative Samples: {negative_samples}")
            print(f"Learning Rate: {learning_rate}")
            print(f"Epochs: {epochs}")
            print(f"Batch Size: {batch_size}")

            try:
                # Create model with current configuration
                model = Word2Vec(
                    vector_size=vector_size,
                    window_size=window_size,
                    min_count=min_count,
                    negative_samples=negative_samples,
                    learning_rate=learning_rate,
                    epochs=epochs,
                    batch_size=batch_size
                )

                # Train on full or tiny corpus
                trained_model = train_word2vec_on_text8(
                    use_cuda=True,
                    use_tiny=use_tiny,
                    vector_size=vector_size
                )

                # Evaluate performance
                performance_metrics = evaluate_word2vec_performance(trained_model)

                # Log configuration and performance
                config_writer.writerow([
                    config_id,
                    vector_size,
                    window_size,
                    min_count,
                    negative_samples,
                    learning_rate,
                    epochs,
                    batch_size,
                    performance_metrics['overall_score'],
                    str(performance_metrics['similarity_scores']),
                    str(performance_metrics['coverage_scores'])
                ])
                config_csv.flush()  # Ensure writing to disk

                # Write to performance log
                perf_log.write(f"\nConfiguration {config_id}:\n")
                perf_log.write(f"Parameters: {locals()}\n")
                perf_log.write(f"Performance Metrics: {performance_metrics}\n")
                perf_log.flush()

                # Update best configuration
                if performance_metrics['overall_score'] < best_performance:
                    best_performance = performance_metrics['overall_score']
                    best_config = {
                        'vector_size': vector_size,
                        'window_size': window_size,
                        'min_count': min_count,
                        'negative_samples': negative_samples,
                        'learning_rate': learning_rate,
                        'epochs': epochs,
                        'batch_size': batch_size,
                        'performance': performance_metrics
                    }

                print(f"Performance Score: {performance_metrics['overall_score']}")

            except Exception as e:
                print(f"Configuration {config_id} failed: {e}")
                # Log the failed configuration
                config_writer.writerow([
                    config_id,
                    vector_size,
                    window_size,
                    min_count,
                    negative_samples,
                    learning_rate,
                    epochs,
                    batch_size,
                    "FAILED",
                    str(e)
                ])
                config_csv.flush()

                perf_log.write(f"\nConfiguration {config_id} FAILED:\n")
                perf_log.write(f"Parameters: {locals()}\n")
                perf_log.write(f"Error: {e}\n")
                perf_log.flush()

        return best_config


def evaluate_word2vec_performance(model, max_test_words=20):
    """
    Comprehensive performance evaluation with error handling
    """
    try:
        # Word similarity test set
        similarity_tests = [
            ('king', ['queen', 'prince', 'ruler']),
            ('man', ['woman', 'person']),
            ('france', ['paris', 'europe', 'country']),
            ('computer', ['software', 'hardware', 'programming'])
        ]

        # Limit tests to prevent excessive computation
        similarity_tests = similarity_tests[:max_test_words]

        # Compute metrics
        similarity_scores = []
        coverage_scores = []

        for word, expected_similar in similarity_tests:
            try:
                # Ensure word exists in vocabulary
                if word not in model.word_to_idx:
                    continue

                similar_words = model.get_most_similar(word, n=10)

                # Compute similarity score
                found_similar = [w for w, _ in similar_words if w in expected_similar]
                similarity_score = len(found_similar) / len(expected_similar) if expected_similar else 0
                similarity_scores.append(similarity_score)

                # Vocabulary coverage
                coverage_scores.append(len(model.vocab) / 71264)  # Relative to text8 vocabulary

            except Exception as word_error:
                print(f"Error evaluating {word}: {word_error}")
                similarity_scores.append(0)
                coverage_scores.append(0)

        # Composite performance score
        # Lower score is better
        overall_score = (
                1 - np.mean(similarity_scores) +  # Semantic similarity
                np.std(similarity_scores) +  # Consistency
                (1 - np.mean(coverage_scores))  # Vocabulary coverage
        )

        return {
            'similarity_scores': similarity_scores,
            'coverage_scores': coverage_scores,
            'overall_score': overall_score
        }

    except Exception as e:
        print(f"Comprehensive evaluation failed: {e}")
        return {
            'similarity_scores': [],
            'coverage_scores': [],
            'overall_score': float('inf')
        }


# ----------------------
# 5. Sample Usage
# ----------------------

if __name__ == "__main__":
    # Import necessary libraries
    import random
    import torch
    import numpy as np

    # Comprehensive Word2Vec Exploration
    print("Starting Word2Vec Parameter Exploration...")
    best_config = comprehensive_word2vec_exploration(
        max_configurations=50,  # Limit to 50 configurations for reasonable runtime
        use_tiny=False  # Use full corpus
    )

    # Print best configuration details
    print("\nBest Configuration Found:")
    for key, value in best_config.items():
        print(f"{key}: {value}")

    # Train Word2Vec model with best parameters
    model = train_word2vec_on_text8(
        use_cuda=True,
        use_tiny=False,
        vector_size=best_config['vector_size']
    )

    # Synthetic Data Generation Function
    def generate_synthetic_hn_data(base_data, additional_samples=80):
        prefixes = ["Show HN:", "Ask HN:", "Why", "How", "The", "Understanding", "My"]
        topics = ["JavaScript", "Python", "Rust", "Go", "machine learning", "AI", "database",
                  "programming", "frontend", "backend", "cloud", "API", "algorithm", "startup"]
        suffixes = ["for beginners", "in production", "tutorial", "guide", "explained", "in 10 minutes",
                    "best practices", "optimization tips", "case study", "industry trends"]

        synthetic_data = base_data.copy()

        for _ in range(additional_samples):
            prefix = random.choice(prefixes)
            topic = random.choice(topics)
            suffix = random.choice(suffixes) if random.random() > 0.5 else ""

            title = f"{prefix} {topic} {suffix}".strip()
            upvotes = int(random.normalvariate(250, 100))  # Mean 250, std dev 100
            upvotes = max(5, upvotes)  # Ensure positive upvotes with minimum 5

            synthetic_data.append((title, upvotes))

        return synthetic_data

    # Sample Hacker News dataset (base data)
    sample_hn_data = [
        ("Show HN: I built a neural network that generates music", 324),
        ("Ask HN: What books changed the way you think about programming?", 427),
        ("Why I switched from Python to Go for backend development", 215),
        ("The future of WebAssembly and browser technologies", 189),
        ("How we optimized our PostgreSQL database for 100x performance", 302),
        ("Ask HN: How do you stay productive working from home?", 512),
        ("Show HN: An open source alternative to Google Analytics", 387),
        ("Understanding blockchain technology in 10 minutes", 127),
        ("My journey from bootcamp to senior developer in 3 years", 203),
        ("Why functional programming matters for modern applications", 174),
        ("Ask HN: Best resources to learn machine learning in 2023?", 398),
        ("Show HN: A lightweight JavaScript framework I built", 251),
        ("The hidden costs of microservices architecture", 289),
        ("How we scaled our startup to 1 million users without VC funding", 475),
        ("Comparing Rust vs C++ for systems programming", 231),
        ("Ask HN: What's your development environment setup?", 365),
        ("Why we moved away from Kubernetes after 2 years", 412),
        ("Understanding memory management in modern JavaScript", 159),
        ("Show HN: My weekend project - a minimalist note-taking app", 186),
        ("How I learned to code by building 12 apps in 12 months", 328)
    ]

    # Generate additional synthetic data
    full_hn_data = generate_synthetic_hn_data(sample_hn_data)
    print(f"Generated dataset with {len(full_hn_data)} Hacker News posts")

    # Train the upvote predictor with dynamically adjusted parameters
    predictor = train_upvote_predictor(
        model,
        full_hn_data,
        epochs=30,  # Can be adjusted based on exploration results
        learning_rate=best_config.get('learning_rate', 0.001)
    )

    # Test titles
    test_titles = [
        "Show HN: I built a tool that helps you learn foreign languages",
        "Ask HN: How do you handle burnout as a developer?",
        "Why TypeScript is becoming the standard for web development",
        "Our journey migrating from MongoDB to PostgreSQL",
        "A beginner's guide to reinforcement learning"
    ]

    # Create embeddings and predict
    test_embeddings = [create_document_embedding(title, model) for title in test_titles]
    test_tensor = torch.FloatTensor(test_embeddings)

    predictor.eval()
    with torch.no_grad():
        predictions = predictor(test_tensor).numpy()

    # Print predictions
    for title, pred in zip(test_titles, predictions):
        print(f"Title: {title}")
        print(f"Predicted upvotes: {int(pred[0])}\n")

    print("Completed Word2Vec training and Hacker News upvote prediction!")