# Improvements to Word2Vec Implementation

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
import matplotlib.pyplot as plt


# ----------------------
# 1. Improved Word2Vec Components
# ----------------------

class Word2VecDataset(Dataset):
    # Original implementation is fine
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
    # Improved negative sampling implementation
    def __init__(self, word2vec_dataset, vocab_size, negative_samples=15):  # Increased negative samples
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

        # Apply power distribution with stronger exponent for better contrast
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
    # Improved skip-gram model with better initialization
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGramNegativeSampling, self).__init__()

        # Embedding matrices
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # Better initialization - using Xavier/Glorot
        nn.init.xavier_uniform_(self.word_embeddings.weight, gain=1.0)
        nn.init.xavier_uniform_(self.context_embeddings.weight, gain=1.0)

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
    def __init__(self, vector_size=300, window_size=5, min_count=5, negative_samples=15,
                 learning_rate=0.0025, epochs=10, batch_size=1024):  # More epochs, smaller learning rate
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

        # Improved optimizer: Adam instead of SGD
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)

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

                    # Gradient clipping to prevent exploding gradients
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)

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

def train_word2vec_on_text8(use_cuda=False, use_tiny=True, vector_size=100):
    """Train Word2Vec on the text8 corpus with improved parameters"""
    # Set up directory
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)

    # Download and prepare text8
    text8_file = download_text8_corpus(data_dir)

    if use_tiny:
        text8_processed = os.path.join(data_dir, "text8_tiny.txt")
        model_file = os.path.join(data_dir, "word2vec_text8_tiny_improved.txt")
        create_tiny_text8(text8_file, text8_processed, max_words=500000)
    else:
        text8_processed = os.path.join(data_dir, "text8_processed.txt")
        model_file = os.path.join(data_dir, "word2vec_text8_improved.txt")

        if not os.path.exists(text8_processed):
            prepare_text8_for_training(text8_file, text8_processed)

    # Create model with improved parameters
    model = Word2Vec(
        vector_size=vector_size,
        window_size=5,
        min_count=5,
        negative_samples=15,  # Increased from 10
        learning_rate=0.001,  # Reduced learning rate for Adam
        epochs=15,  # More epochs
        batch_size=2048  # Larger batch size
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

class HackerNewsPredictor(nn.Module):
    """Improved neural network for predicting Hacker News upvotes"""

    def __init__(self, embedding_dim=100, hidden_dim=256):
        super(HackerNewsPredictor, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),  # Added batch normalization
            nn.Dropout(0.3),  # Increased dropout
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.ReLU()  # Ensure positive predictions
        )

        # Proper weight initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.layers(x)


def create_document_embedding(title, model):
    """Convert a Hacker News title to a document embedding by averaging word vectors"""
    words = model._preprocess_text(title)
    vectors = [model.get_vector(word) for word in words if word in model.word_to_idx]

    if vectors:
        # Average word vectors
        doc_vector = np.mean(vectors, axis=0)
    else:
        doc_vector = np.zeros(model.vector_size)

    return doc_vector


def train_upvote_predictor(model, hn_data, test_size=0.2, epochs=50, learning_rate=0.001):
    """Train a model to predict Hacker News upvotes with improved parameters"""
    # Create document embeddings and prepare data
    X = []
    y = []

    for title, upvotes in hn_data:
        doc_vector = create_document_embedding(title, model)
        X.append(doc_vector)
        y.append(upvotes)

    X = np.array(X)
    y = np.array(y)

    # Normalize target values
    mean_upvotes = np.mean(y)
    std_upvotes = np.std(y)
    print(f"Upvotes statistics: Mean={mean_upvotes:.1f}, Std={std_upvotes:.1f}")

    # Split into train/test sets
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train).unsqueeze(1)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test).unsqueeze(1)

    # Create and train model
    predictor = HackerNewsPredictor(embedding_dim=model.vector_size)
    optimizer = optim.Adam(predictor.parameters(), lr=learning_rate, weight_decay=1e-5)  # Added weight decay
    criterion = nn.MSELoss()

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, verbose=True)

    # Training loop with early stopping
    best_rmse = float('inf')
    patience = 10
    patience_counter = 0
    train_losses = []
    test_losses = []

    for epoch in range(epochs):
        # Forward pass
        predictor.train()
        optimizer.zero_grad()
        outputs = predictor(X_train)
        loss = criterion(outputs, y_train)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Evaluate on test set
        predictor.eval()
        with torch.no_grad():
            test_outputs = predictor(X_test)
            test_loss = criterion(test_outputs, y_test)

            # Calculate RMSE
            train_rmse = torch.sqrt(loss).item()
            test_rmse = torch.sqrt(test_loss).item()

            train_losses.append(train_rmse)
            test_losses.append(test_rmse)

        print(f"Epoch {epoch + 1}/{epochs}, Train RMSE: {train_rmse:.2f}, Test RMSE: {test_rmse:.2f}")

        # Update learning rate
        scheduler.step(test_rmse)

        # Early stopping
        if test_rmse < best_rmse:
            best_rmse = test_rmse
            # Save best model
            torch.save(predictor.state_dict(), "data/best_hn_predictor.pth")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    # Load best model
    predictor.load_state_dict(torch.load("data/best_hn_predictor.pth"))

    # Plot training curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train RMSE')
    plt.plot(test_losses, label='Test RMSE')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True)
    plt.savefig('data/upvote_predictor_training.png')
    plt.close()

    # Final evaluation
    predictor.eval()
    with torch.no_grad():
        train_preds = predictor(X_train).numpy()
        test_preds = predictor(X_test).numpy()

        train_rmse = np.sqrt(np.mean((train_preds - y_train.numpy()) ** 2))
        test_rmse = np.sqrt(np.mean((test_preds - y_test.numpy()) ** 2))

        print(f"\nFinal evaluation:")
        print(f"Train RMSE: {train_rmse:.2f}")
        print(f"Test RMSE: {test_rmse:.2f}")

        # Check if predictions are all positive
        print(f"Min predicted value: {np.min(test_preds):.2f}")
        print(f"Max predicted value: {np.max(test_preds):.2f}")

    return predictor


# ----------------------
# 4. Sample Usage
# ----------------------

if __name__ == "__main__":
    # Train Word2Vec model with improved parameters
    model = train_word2vec_on_text8(use_cuda=True, use_tiny=True, vector_size=100)

    # Sample Hacker News dataset (title, upvotes)
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
    import random

    prefixes = ["Show HN:", "Ask HN:", "Why", "How", "The", "Understanding", "My"]
    topics = ["JavaScript", "Python", "Rust", "Go", "machine learning", "AI", "database",
              "programming", "frontend", "backend", "cloud", "API", "algorithm", "startup"]
    suffixes = ["for beginners", "in production", "tutorial", "guide", "explained", "in 10 minutes",
                "best practices", "optimization tips", "case study", "industry trends"]

    for _ in range(80):  # Add 80 more random entries
        prefix = random.choice(prefixes)
        topic = random.choice(topics)
        suffix = random.choice(suffixes) if random.random() > 0.5 else ""

        title = f"{prefix} {topic} {suffix}".strip()
        upvotes = int(random.normalvariate(250, 100))  # Mean 250, std dev 100
        upvotes = max(5, upvotes)  # Ensure positive upvotes with minimum 5

        sample_hn_data.append((title, upvotes))

    print(f"Generated dataset with {len(sample_hn_data)} Hacker News posts")

    # Train the upvote predictor
    predictor = train_upvote_predictor(model, sample_hn_data, epochs=30)

    # Test on some new titles
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

    for title, pred in zip(test_titles, predictions):
        print(f"Title: {title}")
        print(f"Predicted upvotes: {int(pred[0])}\\n")

    print("Completed Word2Vec training and Hacker News upvote prediction!")