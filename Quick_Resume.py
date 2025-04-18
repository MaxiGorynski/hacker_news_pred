import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import logging
import sys
import os
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
import json
import matplotlib.pyplot as plt


# Make sure this class matches your current implementation
class EnhancedUpvotePredictor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        # Attention mechanism with more sophisticated weighting
        self.attention = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Softmax(dim=1)
        )

        # Main network with enhanced regularization and depth
        self.main_network = nn.Sequential(
            # Layer 1 - Expanded input layer
            nn.Linear(input_dim, 768),
            nn.BatchNorm1d(768),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.6),

            # Layer 2 - Deeper network with increased width
            nn.Linear(768, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),

            # Layer 3 - Additional complexity
            nn.Linear(512, 384),
            nn.BatchNorm1d(384),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.45),

            # Layer 4 - Continued depth
            nn.Linear(384, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4),

            # Layer 5 - Further refinement
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            # Final layer with explicit initialization
            nn.Linear(128, 1)
        )

    def forward(self, x):
        # Apply sophisticated attention mechanism
        attention_weights = self.attention(x)

        # Element-wise multiplication with residual connection
        attended_input = x * attention_weights
        enhanced_input = x + attended_input

        # Pass through main network
        return self.main_network(enhanced_input)


# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


# This is the quick resume function that assumes X and y are already in memory
def resume_training_from_memory(X, y, test_size=0.2):
    """
    Resume training using X and y variables already in memory

    Args:
        X: Embeddings array (numpy array)
        y: Target values array (numpy array)
        test_size: Fraction to use for testing

    Returns:
        predictor, y_mean, y_std
    """
    logger.info(f"Resuming training with existing X and y (shapes: {X.shape}, {y.shape})")

    # Robust log transform and normalization
    y_log = np.log1p(y)
    y_mean = np.mean(y_log)
    y_std = np.std(y_log)
    y_normalized = (y_log - y_mean) / y_std

    logger.info(f"Upvote statistics - Mean: {y_mean:.4f}, Std: {y_std:.4f}")

    # Data splitting
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_normalized, test_size=test_size, random_state=42
    )
    logger.info(f"Train set: {X_train.shape[0]}, Test set: {X_test.shape[0]}")

    # Save a backup of the data for safety
    try:
        np.save('X_embeddings_backup.npy', X)
        np.save('y_values_backup.npy', y)
        logger.info("Created backup of embeddings")
    except Exception as e:
        logger.warning(f"Could not save backup: {e}")

    # Convert to tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train).unsqueeze(1)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test).unsqueeze(1)

    # Create predictor
    logger.info(f"Initializing enhanced predictor with input dim: {X_train.shape[1]}")
    predictor = EnhancedUpvotePredictor(X_train.shape[1])

    # Optimizer setup
    optimizer = optim.AdamW(
        predictor.parameters(),
        lr=0.0005,
        weight_decay=0.03,
        betas=(0.9, 0.999)
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # Training parameters
    patience = 15
    best_rmse = float('inf')
    patience_counter = 0
    max_epochs = 50  # Shorter training to get results faster

    # Tracking metrics
    train_losses, test_losses = [], []
    train_rmses, test_rmses = [], []
    prediction_stds = []
    spearman_corrs = []

    # Mini-batch training
    batch_size = 128
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    logger.info(f"Starting Upvote Predictor Training Loop")

    # Training loop
    for epoch in tqdm(range(max_epochs), desc="Training epochs"):
        # Training phase
        predictor.train()
        epoch_loss = 0.0
        num_batches = 0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()

            # Forward pass
            outputs = predictor(X_batch)

            # Loss calculation
            loss = nn.functional.mse_loss(outputs, y_batch)

            # Backward pass
            loss.backward()

            # Gradient clipping
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
            test_loss = nn.functional.mse_loss(test_outputs, y_test)
            test_losses.append(test_loss.item())

            # Denormalize predictions
            train_outputs = predictor(X_train)
            train_preds_orig = np.expm1(train_outputs.numpy() * y_std + y_mean)
            train_true_orig = np.expm1(y_train.numpy() * y_std + y_mean)
            test_preds_orig = np.expm1(test_outputs.numpy() * y_std + y_mean)
            test_true_orig = np.expm1(y_test.numpy() * y_std + y_mean)

            # Calculate metrics
            train_rmse = np.sqrt(np.mean((train_preds_orig - train_true_orig) ** 2))
            test_rmse = np.sqrt(np.mean((test_preds_orig - test_true_orig) ** 2))
            pred_std = np.std(test_preds_orig)

            train_rmses.append(train_rmse)
            test_rmses.append(test_rmse)
            prediction_stds.append(pred_std)

            # Rank correlation
            corr, _ = spearmanr(test_preds_orig.flatten(), test_true_orig.flatten())
            spearman_corrs.append(corr)

            # Progress logging
            logger.info(f"Epoch {epoch + 1}/{max_epochs}")
            logger.info(f"Train Loss: {avg_train_loss:.6f}, Test Loss: {test_loss.item():.6f}")
            logger.info(f"Train RMSE: {train_rmse:.2f}, Test RMSE: {test_rmse:.2f}")
            logger.info(f"Prediction Std Dev: {pred_std:.4f}")
            logger.info(f"Spearman Correlation: {corr:.4f}")

            # Update learning rate
            scheduler.step(test_loss.item())

            # Model checkpoint
            if test_rmse < best_rmse:
                improvement = (best_rmse - test_rmse) / best_rmse * 100 if best_rmse != float('inf') else 100
                best_rmse = test_rmse
                patience_counter = 0
                torch.save(predictor.state_dict(), 'best_predictor_model.pth')
                logger.info(f"New best model saved with RMSE: {best_rmse:.2f} (improved by {improvement:.2f}%)")
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

    # Load best model
    predictor.load_state_dict(torch.load('best_predictor_model.pth'))
    logger.info("Loaded best model")

    # Save normalization parameters
    normalization_params = {
        'y_mean': float(y_mean),
        'y_std': float(y_std)
    }
    with open('normalization_params.json', 'w') as f:
        json.dump(normalization_params, f)
    logger.info("Saved normalization parameters for inference")

    logger.info("Training completed successfully")
    return predictor, y_mean, y_std


# Run this if you have X and y in memory
if __name__ == "__main__":
    # These lines assume X and y are already defined in the global scope
    # If they're not, you'll need to modify this script

    # Get X and y from the global namespace
    import sys

    this_module = sys.modules[__name__]

    # Try to get X and y from globals
    if 'X' in globals() and 'y' in globals():
        X = globals()['X']
        y = globals()['y']
        logger.info(f"Found X and y in globals with shapes {X.shape} and {y.shape}")

        # Run training
        predictor, y_mean, y_std = resume_training_from_memory(X, y)
    else:
        logger.error("Could not find X and y in globals! Make sure they're defined before running this script.")