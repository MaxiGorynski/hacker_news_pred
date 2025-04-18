import torch
import torch.nn as nn
import numpy as np
import re
import pandas as pd
from tqdm import tqdm
import sys
import os
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


# Define the EnhancedUpvotePredictor class
# Update just the model architecture to match what was trained
class EnhancedUpvotePredictor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        # Attention mechanism for input features
        self.attention = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, input_dim),  # Changed from 1 to input_dim
            nn.Softmax(dim=1)
        )

        # Main network with increased width and depth
        self.main_network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),

            nn.Linear(512, 384),
            nn.BatchNorm1d(384),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),

            nn.Linear(384, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.45),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(64, 1),
        )

    def forward(self, x):
        # Apply attention mechanism (with correct dimensions)
        attention_weights = self.attention(x)  # Simplified based on correct dimensions

        # Element-wise multiplication with attention weights
        attended_input = x * attention_weights

        # Residual connection
        enhanced_input = x + attended_input

        # Pass through main network
        return self.main_network(enhanced_input)


def basic_predict_upvotes(title, model):
    """A simple fallback predictor that ignores the neural network."""
    words = model._preprocess_text(title)

    # Base score
    score = 3

    # Length bonus
    if len(words) > 10:
        score += 2
    elif len(words) > 5:
        score += 1

    # Keyword bonuses
    if 'show' in words:
        score += 3
    if 'ask' in words:
        score += 1

    # Topic bonuses
    tech_words = ['ai', 'blockchain', 'crypto', 'cloud', 'ml', 'neural', 'quantum']
    if any(word in tech_words for word in words):
        score += 4

    return max(1, min(50, score))  # Clamp between 1 and 50

# Define the Word2Vec class for loading embeddings
class Word2Vec:
    def __init__(self, vector_size=100):
        self.vector_size = vector_size
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.vocab = []

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

    def load(self, filename):
        """Load pre-trained word vectors"""
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.vocab = []

        with open(filename, 'r', encoding='utf-8') as f:
            header = f.readline().split()
            vocab_size = int(header[0])
            self.vector_size = int(header[1])

            # Store embeddings as numpy arrays
            self.embeddings = np.zeros((vocab_size, self.vector_size))

            for i, line in enumerate(f):
                parts = line.rstrip().split(' ')
                word = parts[0]

                self.word_to_idx[word] = i
                self.idx_to_word[i] = word
                self.vocab.append(word)

                self.embeddings[i] = np.array([float(x) for x in parts[1:]])

        logger.info(f"Loaded {len(self.vocab)} words with dimension {self.vector_size}")
        return self

    def get_vector(self, word):
        """Get vector for a word"""
        if word in self.word_to_idx:
            idx = self.word_to_idx[word]
            return self.embeddings[idx]
        return None


def find_file(filename, search_paths=None):
    """Find a file by searching in multiple locations"""
    if search_paths is None:
        # Default search paths
        search_paths = [
            '.',  # Current directory
            '..',  # Parent directory
            '/Users/supriyarai/Code/MLXercises',  # Home project directory
            os.path.join(os.path.dirname(__file__), '..'),  # Parent of script directory
        ]

    # Try each path
    for path in search_paths:
        full_path = os.path.join(path, filename)
        if os.path.exists(full_path):
            logger.info(f"Found {filename} at {full_path}")
            return full_path

    # If not found
    raise FileNotFoundError(f"Could not find {filename} in any search path")


# Function to create document embeddings
def create_document_embedding(title, model, upvotes=None):
    """Create embedding for a document with additional features"""
    words = model._preprocess_text(title)

    vectors = []
    word_weights = []

    for word in words:
        vec = model.get_vector(word)
        if vec is not None:
            # Position-based weighting
            position_weight = 1.0 / (abs(words.index(word) - len(words) / 2) + 1)
            unique_word_weight = 1.0 / (words.count(word) ** 0.5)

            vectors.append(vec)
            word_weights.append(position_weight * unique_word_weight)

    if vectors:
        # Weighted average embedding
        doc_vector = np.average(vectors, axis=0, weights=word_weights)

        # Additional features - match exactly what was used during training
        features = [
            len(words),  # Title length
            sum(1 for word in words if word.isupper()),  # Uppercase word count
            sum(1 for word in ['show', 'ask', 'tell'] if word in words),  # HN post type
            np.mean([len(word) for word in words]) if words else 0,  # Average word length
            # Add 2 more features that were in the training model - based on the 106 input size
            1.0,  # Placeholder for the 5th feature (likely log upvotes during training)
            1.0  # Placeholder for the 6th feature (likely binary upvote indicator)
        ]

        # Concatenate embedding and features
        return np.concatenate([doc_vector, features])
    else:
        # No valid vectors found
        return np.zeros(model.vector_size + 6)  # Must be 106 total (100 + 6 features)


# Function to load real titles and scores
def load_hn_data(csv_path, limit=1000):
    """Load a sample of Hacker News titles and scores for comparison"""
    try:
        df = pd.read_csv(csv_path)
        if limit:
            df = df.sample(n=min(limit, len(df)))
        return dict(zip(df['title'], df['upvotes']))
    except Exception as e:
        logger.error(f"Error loading HN data: {e}")
        return {}


# Function to predict upvotes for a title
# 1. First, add debugging to see what's going wrong
def predict_upvotes(title, model, predictor, y_mean, y_std):
    # Create document embedding
    doc_vector = create_document_embedding(title, model)

    # Convert to tensor
    X = torch.FloatTensor(doc_vector).unsqueeze(0)

    # Print intermediate values for debugging
    predictor.eval()
    with torch.no_grad():
        normalized_prediction = predictor(X).item()
        print(f"DEBUG - Normalized prediction: {normalized_prediction:.4f}")
        print(f"DEBUG - y_mean: {y_mean:.4f}, y_std: {y_std:.4f}")

        # Raw prediction before rounding
        raw_prediction = np.expm1(normalized_prediction * y_std + y_mean)
        print(f"DEBUG - Raw prediction: {raw_prediction:.4f}")

    return max(0, int(round(raw_prediction)))


def train_simple_model(data_path, limit=10000, epochs=10):
    # Load data
    df = pd.read_csv(data_path)
    if limit:
        df = df.sample(n=min(limit, len(df)), random_state=42)

    # Get features and targets
    w2v_model = Word2Vec().load('HN_Corpus_Model_Weights.txt')
    X = np.array([create_document_embedding(title, w2v_model) for title in tqdm(df['title'])])
    y = np.log1p(df['upvotes'].values)

    # Create model
    model = SimpleUpvotePredictor(X.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Train
    for epoch in range(epochs):
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y).unsqueeze(1)

        model.train()
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()

        logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

    return model, np.mean(y), np.std(y)

# Main function
# Main function
def main():
    try:
        # Dynamically find the files
        word2vec_path = find_file('HN_Corpus_Model_Weights.txt')
        predictor_path = find_file('best_predictor_model.pth')
        hn_data_path = find_file('df_200K.csv')

        logger.info("Loading models...")

        # Load Word2Vec model
        try:
            w2v_model = Word2Vec()
            w2v_model.load(word2vec_path)
            logger.info(f"Word2Vec model loaded with {len(w2v_model.vocab)} words")
        except Exception as e:
            logger.error(f"Error loading Word2Vec model: {e}")
            return

        # Load upvote predictor - FIXED: Use exactly 106 as input dimension
        try:
            # Create predictor with the exact input dimension from the trained model
            input_dim = 106  # Fix this to match the trained model (100 for embedding + 6 features)
            predictor = EnhancedUpvotePredictor(input_dim)

            # Load weights
            predictor.load_state_dict(torch.load(predictor_path))
            logger.info("Upvote predictor model loaded")

            # Verify model weights - add this section
            logger.info("Verifying model weights:")
            for name, param in predictor.named_parameters():
                if param.requires_grad:
                    zeros = (param == 0).sum().item()
                    total = param.numel()
                    logger.info(
                        f"  Layer {name}: shape {param.shape}, zeros: {zeros}/{total} ({zeros / total * 100:.1f}%)")
                    # Print min, max, and mean to check for reasonable values
                    if param.numel() > 0:
                        logger.info(
                            f"    Stats: min={param.min().item():.4f}, max={param.max().item():.4f}, mean={param.mean().item():.4f}")

            # Test model on a fixed input - add this section
            logger.info("Testing model with a fixed input:")
            test_title = "Test title with some words"
            test_vector = create_document_embedding(test_title, w2v_model)
            test_input = torch.FloatTensor(test_vector).unsqueeze(0)

            # Check tensor shapes
            logger.info(f"  Input shape: {test_input.shape}")

            # Get model output for each layer
            predictor.eval()
            with torch.no_grad():
                # Check attention weights
                attention_weights = predictor.attention(test_input)
                logger.info(f"  Attention output shape: {attention_weights.shape}")
                logger.info(
                    f"  Attention stats: min={attention_weights.min().item():.6f}, max={attention_weights.max().item():.6f}")

                # Test full model
                output = predictor(test_input)
                logger.info(f"  Model output: {output.item():.6f}")

        except Exception as e:
            logger.error(f"Error loading predictor model: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return

        # Try a few different normalization parameter combinations
        y_mean_options = [1.5, 1.65, 1.8]
        y_std_options = [1.0, 1.25, 1.5]

        print("\n===== Testing different normalization parameters =====")
        test_title = "Show HN: A new framework for building web applications"

        for mean in y_mean_options:
            for std in y_std_options:
                X = torch.FloatTensor(create_document_embedding(test_title, w2v_model)).unsqueeze(0)
                with torch.no_grad():
                    norm_pred = predictor(X).item()
                    raw_pred = np.expm1(norm_pred * std + mean)
                    print(f"Mean={mean:.2f}, Std={std:.2f}: Prediction={int(round(raw_pred))}")

        # Load a larger HN sample for more accurate normalization parameters
        real_scores = load_hn_data(hn_data_path, limit=10000)  # Increase from 1000 to 10000
        logger.info(f"Loaded {len(real_scores)} real HN titles for comparison")

        predictor.load_state_dict(torch.load(predictor_path))
        logger.info("Upvote predictor model loaded")

        # Add the diagnostic code here
        logger.info("Examining saved model state dictionary directly:")
        saved_model = torch.load(predictor_path)
        logger.info(f"Keys in saved model: {list(saved_model.keys())}")

        # Check for zero weights in key layers
        final_layer_key = 'main_network.20.weight'  # The final layer before output
        if final_layer_key in saved_model:
            weights = saved_model[final_layer_key]
            zeros = (weights == 0).sum().item()
            total = weights.numel()
            logger.info(f"Final layer weights: zeros {zeros}/{total} ({zeros / total * 100:.1f}%)")
            logger.info(f"Final layer weight values: {weights.flatten().tolist()}")

        # Use these parameters based on testing
        y_mean = 1.5  # Adjust based on the results above
        y_std = 1.0  # Adjust based on the results above
        logger.info(f"Using normalization parameters: mean={y_mean:.4f}, std={y_std:.4f}")

        # Interactive prediction
        print("\n===== Hacker News Upvote Predictor =====")
        print("Enter a title to predict upvotes, or 'q' to quit")
        print("Enter 'r' to see a random real title and its score")

        while True:
            print("\n" + "-" * 50)
            user_input = input("Enter a title (or 'q' to quit, 'r' for random): ")

            if user_input.lower() == 'q':
                break

            if user_input.lower() == 'r':
                # Get a random real title
                random_title = np.random.choice(list(real_scores.keys()))
                real_score = real_scores[random_title]

                print(f"\nRandom title: \"{random_title}\"")
                print(f"Real score: {real_score}")

                # Predict score for comparison
                predicted_score = predict_upvotes(random_title, w2v_model, predictor, y_mean, y_std)
                print(f"Predicted score: {predicted_score}")

                # Calculate error
                error_pct = abs(predicted_score - real_score) / max(1, real_score) * 100
                print(f"Error: {error_pct:.1f}%")

            else:
                # User entered a custom title
                title = user_input
                predicted_score = predict_upvotes(title, w2v_model, predictor, y_mean, y_std)

                print(f"\nTitle: \"{title}\"")
                print(f"Predicted score: {predicted_score}")

                # Check if this is similar to a real title we have
                most_similar = None
                highest_sim = -1

                for real_title in real_scores:
                    # Simple similarity measure: word overlap
                    title_words = set(w2v_model._preprocess_text(title))
                    real_words = set(w2v_model._preprocess_text(real_title))

                    if title_words and real_words:
                        overlap = len(title_words.intersection(real_words)) / len(title_words.union(real_words))
                        if overlap > highest_sim and overlap > 0.5:  # At least 50% similarity
                            highest_sim = overlap
                            most_similar = real_title

                if most_similar:
                    print(f"\nSimilar real title: \"{most_similar}\"")
                    print(f"Real score: {real_scores[most_similar]}")
                    print(f"Similarity: {highest_sim:.1%}")

    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        import traceback
        logger.error(traceback.format_exc())


#Copy of main() repurposed to respond directly to single input
def call_and_response(user_input: str)-> int:
    try:
        # Dynamically find the files
        word2vec_path = find_file('HN_Corpus_Model_Weights.txt')
        predictor_path = find_file('best_predictor_model.pth')
        hn_data_path = find_file('df_200K.csv')

        logger.info("Loading models...")

        # Load Word2Vec model
        try:
            w2v_model = Word2Vec()
            w2v_model.load(word2vec_path)
            logger.info(f"Word2Vec model loaded with {len(w2v_model.vocab)} words")
        except Exception as e:
            logger.error(f"Error loading Word2Vec model: {e}")
            return

        # Load upvote predictor - FIXED: Use exactly 106 as input dimension
        try:
            # Create predictor with the exact input dimension from the trained model
            input_dim = 106  # Fix this to match the trained model (100 for embedding + 6 features)
            predictor = EnhancedUpvotePredictor(input_dim)

            # Load weights
            predictor.load_state_dict(torch.load(predictor_path))
            logger.info("Upvote predictor model loaded")
        except Exception as e:
            logger.error(f"Error loading predictor model: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return

        # Load sample HN data for comparison
        real_scores = load_hn_data(hn_data_path)
        logger.info(f"Loaded {len(real_scores)} real HN titles for comparison")

        # Calculate log transform parameters from the data
        upvotes = np.array(list(real_scores.values()))
        log_upvotes = np.log1p(upvotes)
        y_mean = np.mean(log_upvotes)
        y_std = np.std(log_upvotes)
        logger.info(f"Calculated normalization parameters: mean={y_mean:.4f}, std={y_std:.4f}")

        title = user_input
        predicted_score = predict_upvotes(title, w2v_model, predictor, y_mean, y_std)
            
        print(f"\nTitle: \"{title}\"")
        print(f"Predicted score: {predicted_score}")

        # Check if this is similar to a real title we have
        most_similar = None
        highest_sim = -1

        for real_title in real_scores:
        # Simple similarity measure: word overlap
            title_words = set(w2v_model._preprocess_text(title))
            real_words = set(w2v_model._preprocess_text(real_title))

            if title_words and real_words:
                overlap = len(title_words.intersection(real_words)) / len(title_words.union(real_words))
                if overlap > highest_sim and overlap > 0.5:  # At least 50% similarity
                    highest_sim = overlap
                    most_similar = real_title

        if most_similar:
            print(f"\nSimilar real title: \"{most_similar}\"")
            print(f"Real score: {real_scores[most_similar]}")
            print(f"Similarity: {highest_sim:.1%}")
        
        return predicted_score

    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()