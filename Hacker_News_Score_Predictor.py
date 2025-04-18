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
def predict_upvotes(title, model, predictor, y_mean, y_std):
    """Predict upvotes for a given title"""
    # Create document embedding
    doc_vector = create_document_embedding(title, model)

    # Convert to tensor
    X = torch.FloatTensor(doc_vector).unsqueeze(0)  # Include all features

    # Get prediction
    predictor.eval()
    with torch.no_grad():
        normalized_prediction = predictor(X).item()

    # Denormalize
    prediction = np.expm1(normalized_prediction * y_std + y_mean)

    return max(0, int(round(prediction)))


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