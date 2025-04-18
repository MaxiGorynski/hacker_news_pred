import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from tqdm import tqdm

# Define simplified models for each feature group
class AuthorModel(nn.Module):
    def __init__(self, input_dim):
        super(AuthorModel, self).__init__()
        self.regressor = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )
    
    def forward(self, x):
        return self.regressor(x)

class DomainModel(nn.Module):
    def __init__(self, input_dim):
        super(DomainModel, self).__init__()
        self.regressor = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )
    
    def forward(self, x):
        return self.regressor(x)

class TemporalModel(nn.Module):
    def __init__(self, input_dim):
        super(TemporalModel, self).__init__()
        self.regressor = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )
    
    def forward(self, x):
        return self.regressor(x)

class FusionModel(nn.Module):
    def __init__(self, num_models=3):
        super(FusionModel, self).__init__()
        self.fusion_layer = nn.Sequential(
            nn.Linear(num_models, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 1)
        )
    
    def forward(self, x):
        return self.fusion_layer(x)

class HNLateFusionTrainer:
    def __init__(self, df, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the late fusion trainer with simplified feature sets
        
        Args:
            df: Processed dataframe with features
            device: Computation device
        """
        self.df = df
        self.device = device
        self.scalers = {}
        self.models = {}
        self.histories = {}
        
        # Simplified feature sets
        self.author_features = [
            'author_avg_upvotes', 'author_median_upvotes', 'author_max_upvotes'
        ]
        
        self.domain_features = [
            'domain_avg_upvotes', 'domain_median_upvotes', 'domain_max_upvotes'
        ]
        
        self.temporal_features = [
            'post_hour', 'weekday_encoded'
        ]
        
        # Check if features exist in dataframe, remove if not
        all_features = set(df.columns)
        self.author_features = [f for f in self.author_features if f in all_features]
        self.domain_features = [f for f in self.domain_features if f in all_features]
        self.temporal_features = [f for f in self.temporal_features if f in all_features]
        
        # Log transform the target variable to address skewness
        print("Applying log transform to target variable")
        self.df['upvotes_log'] = np.log1p(self.df['upvotes'])
        
        # Pre-process high-variance features
        print("Pre-processing high-variance features")
        self._preprocess_high_variance_features()
        
        # Target variable
        self.target = 'upvotes_log'  # Use log-transformed target
    
    def _preprocess_high_variance_features(self):
        """Log-transform features with high variance to stabilize training"""
        for col in ['author_max_upvotes', 'domain_max_upvotes', 'author_avg_upvotes', 'domain_avg_upvotes']:
            if col in self.df.columns:
                # Add a small constant to handle zeros
                self.df[f'{col}_log'] = np.log1p(self.df[col])
                
                # Update feature lists
                if col in self.author_features:
                    self.author_features.remove(col)
                    self.author_features.append(f'{col}_log')
                elif col in self.domain_features:
                    self.domain_features.remove(col)
                    self.domain_features.append(f'{col}_log')
    
    def validate_data(self):
        """Check data for NaN, infinity, or extreme values"""
        for feature_group, features in [
            ('author', self.author_features),
            ('domain', self.domain_features),
            ('temporal', self.temporal_features)
        ]:
            data = self.df[features]
            print(f"\n{feature_group.capitalize()} features:")
            print(f"NaN count: {data.isna().sum().sum()}")
            print(f"Inf count: {np.isinf(data.values).sum()}")
            print(f"Min values: \n{data.min()}")
            print(f"Max values: \n{data.max()}")
        
        # Check target variable
        print(f"\nTarget variable ({self.target}):")
        print(f"NaN count: {self.df[self.target].isna().sum()}")
        print(f"Min: {self.df[self.target].min()}, Max: {self.df[self.target].max()}")
        print(f"Mean: {self.df[self.target].mean()}, Std: {self.df[self.target].std()}")
        
    def prepare_data(self):
        """Prepare all datasets for training"""
        # Validate data first
        self.validate_data()
        
        # Replace infinite values with NaN
        self.df = self.df.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN values for all feature columns with median
        for col in self.df.select_dtypes(include=[np.number]).columns:
            if self.df[col].isna().any():
                median_val = self.df[col].median()
                self.df[col] = self.df[col].fillna(median_val)
        
        # Split data
        train_df, test_df = train_test_split(self.df, test_size=0.2, random_state=42)
        
        # Prepare datasets for each model
        print("Preparing author dataset...")
        author_train_loader, author_val_loader = self._prepare_feature_data(
            train_df, test_df, self.author_features, 'author'
        )
        
        print("Preparing domain dataset...")
        domain_train_loader, domain_val_loader = self._prepare_feature_data(
            train_df, test_df, self.domain_features, 'domain'
        )
        
        print("Preparing temporal dataset...")
        temporal_train_loader, temporal_val_loader = self._prepare_feature_data(
            train_df, test_df, self.temporal_features, 'temporal'
        )
        
        self.loaders = {
            'author': (author_train_loader, author_val_loader),
            'domain': (domain_train_loader, domain_val_loader),
            'temporal': (temporal_train_loader, temporal_val_loader),
        }
        
        # Store data for fusion model training later
        self.train_df = train_df
        self.test_df = test_df
    
    def _prepare_feature_data(self, train_df, test_df, feature_list, name):
        """Prepare dataset for a feature group with robust preprocessing"""
        try:
            # Extract features
            X_train = train_df[feature_list].copy()
            y_train = train_df[self.target].values
            
            X_test = test_df[feature_list].copy()
            y_test = test_df[self.target].values
            
            # Check for and handle infinite values
            X_train = X_train.replace([np.inf, -np.inf], np.nan)
            X_test = X_test.replace([np.inf, -np.inf], np.nan)
            
            # Handle missing values - use median
            for col in X_train.columns:
                median_val = X_train[col].median()
                X_train[col] = X_train[col].fillna(median_val)
                X_test[col] = X_test[col].fillna(median_val)
            
            # More conservative outlier handling (5th and 95th percentiles)
            for col in X_train.columns:
                q5 = X_train[col].quantile(0.05)
                q95 = X_train[col].quantile(0.95)
                X_train[col] = X_train[col].clip(q5, q95)
                X_test[col] = X_test[col].clip(q5, q95)
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Double check for NaN or Inf values after scaling
            if np.isnan(X_train_scaled).any() or np.isinf(X_train_scaled).any():
                print(f"Warning: NaN or Inf values in {name} training data after scaling")
                X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0, posinf=0.0, neginf=0.0)
            
            if np.isnan(X_test_scaled).any() or np.isinf(X_test_scaled).any():
                print(f"Warning: NaN or Inf values in {name} test data after scaling")
                X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Store scaler for inference
            self.scalers[name] = scaler
            
            # Convert to tensors
            train_data = TensorDataset(
                torch.FloatTensor(X_train_scaled), 
                torch.FloatTensor(y_train.reshape(-1, 1))
            )
            test_data = TensorDataset(
                torch.FloatTensor(X_test_scaled), 
                torch.FloatTensor(y_test.reshape(-1, 1))
            )
            
            # Create data loaders
            train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
            test_loader = DataLoader(test_data, batch_size=64, shuffle=False)
            
            return train_loader, test_loader
        except Exception as e:
            print(f"Error preparing {name} data: {e}")
            missing_features = [f for f in feature_list if f not in train_df.columns]
            if missing_features:
                print(f"Missing features: {missing_features}")
            raise
    
    def train_model(self, model_name, model, train_loader, val_loader, num_epochs=15, early_stopping=10):
        """Train a model with early stopping and gradient clipping"""
        print(f"Training {model_name} model...")
        model = model.to(self.device)
        
        # Use Huber loss instead of MSE for robustness to outliers
        criterion = nn.HuberLoss(delta=1.0)
        optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Reduced learning rate
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
        
        best_val_loss = float('inf')
        patience_counter = 0
        history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            train_losses = []
            
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Check input data
                if torch.isnan(inputs).any() or torch.isinf(inputs).any():
                    print(f"Warning: NaN or Inf detected in inputs for {model_name}")
                    inputs = torch.nan_to_num(inputs)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                
                # Check for NaN in outputs
                if torch.isnan(outputs).any():
                    print("NaN detected in outputs, skipping batch")
                    continue

                loss = criterion(outputs, targets)

                if torch.isnan(loss).any():
                    print("NaN detected in loss, skipping batch")
                    continue

                loss.backward()
                
                # Tighter gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()

                train_losses.append(loss.item())
            
            # Validation phase
            model.eval()
            val_losses = []
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_losses.append(loss.item())
            
            train_loss = np.mean(train_losses)
            val_loss = np.mean(val_losses)
            
            # Check for numeric issues in loss
            if np.isnan(train_loss) or np.isinf(train_loss):
                print(f"Warning: Train loss is {train_loss} at epoch {epoch+1}")
            if np.isnan(val_loss) or np.isinf(val_loss):
                print(f"Warning: Val loss is {val_loss} at epoch {epoch+1}")
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            
            print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            
            # Update learning rate based on validation loss
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                self.models[model_name] = model
            else:
                patience_counter += 1
                if patience_counter >= early_stopping:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break
        
        self.histories[model_name] = history
        return model, history
    
    def train_all_models(self):
        """Train all individual models"""
        # Train author model
        author_model = AuthorModel(input_dim=len(self.author_features))
        author_train_loader, author_val_loader = self.loaders['author']
        self.train_model('author', author_model, author_train_loader, author_val_loader)
        
        # Train domain model
        domain_model = DomainModel(input_dim=len(self.domain_features))
        domain_train_loader, domain_val_loader = self.loaders['domain']
        self.train_model('domain', domain_model, domain_train_loader, domain_val_loader)
        
        # Train temporal model
        temporal_model = TemporalModel(input_dim=len(self.temporal_features))
        temporal_train_loader, temporal_val_loader = self.loaders['temporal']
        self.train_model('temporal', temporal_model, temporal_train_loader, temporal_val_loader)
    
    def generate_predictions(self, df, model_name):
        """Generate predictions for a specific model"""
        model = self.models[model_name]
        model.eval()
        
        with torch.no_grad():
            # Get features for the model
            if model_name == 'author':
                features = self.author_features
            elif model_name == 'domain':
                features = self.domain_features
            elif model_name == 'temporal':
                features = self.temporal_features
            
            X = df[features].values
            X_scaled = self.scalers[model_name].transform(X)
            
            # Safety check
            X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
            
            inputs = torch.FloatTensor(X_scaled).to(self.device)
            
            # Generate predictions
            outputs = model(inputs)
            return outputs.cpu().numpy().flatten()
    
    def prepare_fusion_data(self):
        """Prepare data for training the fusion model"""
        print("Generating predictions from individual models...")
        
        # Generate predictions for train data
        train_author_preds = self.generate_predictions(self.train_df, 'author')
        train_domain_preds = self.generate_predictions(self.train_df, 'domain')
        train_temporal_preds = self.generate_predictions(self.train_df, 'temporal')
        
        # Generate predictions for test data
        test_author_preds = self.generate_predictions(self.test_df, 'author')
        test_domain_preds = self.generate_predictions(self.test_df, 'domain')
        test_temporal_preds = self.generate_predictions(self.test_df, 'temporal')
        
        # Combine predictions
        train_fusion_inputs = np.column_stack([
            train_author_preds, train_domain_preds, train_temporal_preds
        ])
        
        test_fusion_inputs = np.column_stack([
            test_author_preds, test_domain_preds, test_temporal_preds
        ])
        
        # Check for any NaN/Inf and fix them
        train_fusion_inputs = np.nan_to_num(train_fusion_inputs, nan=0.0, posinf=0.0, neginf=0.0)
        test_fusion_inputs = np.nan_to_num(test_fusion_inputs, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Create TensorDatasets
        train_targets = self.train_df[self.target].values
        test_targets = self.test_df[self.target].values
        
        train_fusion_data = TensorDataset(
            torch.FloatTensor(train_fusion_inputs),
            torch.FloatTensor(train_targets.reshape(-1, 1))
        )
        
        test_fusion_data = TensorDataset(
            torch.FloatTensor(test_fusion_inputs),
            torch.FloatTensor(test_targets.reshape(-1, 1))
        )
        
        # Create data loaders
        train_fusion_loader = DataLoader(train_fusion_data, batch_size=64, shuffle=True)
        test_fusion_loader = DataLoader(test_fusion_data, batch_size=64, shuffle=False)
        
        return train_fusion_loader, test_fusion_loader
    
    def train_fusion_model(self):
        """Train the fusion model"""
        print("Training fusion model...")
        
        # Prepare fusion data
        train_fusion_loader, test_fusion_loader = self.prepare_fusion_data()
        
        # Create and train fusion model
        fusion_model = FusionModel(num_models=3)
        self.train_model('fusion', fusion_model, train_fusion_loader, test_fusion_loader)
    
    def evaluate(self):
        """Evaluate all models on test data"""
        results = {}
        
        # Individual models evaluation
        for model_name in ['author', 'domain', 'temporal']:
            preds = self.generate_predictions(self.test_df, model_name)
            
            # First evaluate on log scale (what model was trained on)
            log_true_values = self.test_df[self.target].values
            log_mse = np.mean((preds - log_true_values) ** 2)
            log_rmse = np.sqrt(log_mse)
            log_mae = np.mean(np.abs(preds - log_true_values))
            
            # Convert back to original scale for interpretability
            preds_original = np.expm1(preds)
            true_values = self.test_df['upvotes'].values
            
            # Calculate metrics on original scale
            mse = np.mean((preds_original - true_values) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(preds_original - true_values))
            
            results[model_name] = {
                'RMSE': rmse, 'MAE': mae, 
                'Log_RMSE': log_rmse, 'Log_MAE': log_mae
            }
        
        # Fusion model evaluation
        fusion_preds = self.predict_with_fusion(self.test_df)
        
        # Evaluate on log scale
        log_true_values = self.test_df[self.target].values
        fusion_log_mse = np.mean((fusion_preds - log_true_values) ** 2)
        fusion_log_rmse = np.sqrt(fusion_log_mse)
        fusion_log_mae = np.mean(np.abs(fusion_preds - log_true_values))
        
        # Convert to original scale for interpretability
        fusion_preds_original = np.expm1(fusion_preds)
        true_values = self.test_df['upvotes'].values
        
        fusion_mse = np.mean((fusion_preds_original - true_values) ** 2)
        fusion_rmse = np.sqrt(fusion_mse)
        fusion_mae = np.mean(np.abs(fusion_preds_original - true_values))
        
        results['fusion'] = {
            'RMSE': fusion_rmse, 'MAE': fusion_mae,
            'Log_RMSE': fusion_log_rmse, 'Log_MAE': fusion_log_mae
        }
        
        return results
    
    def predict_with_fusion(self, df):
        """Make predictions using the fusion model"""
        # Generate predictions from individual models
        author_preds = self.generate_predictions(df, 'author')
        domain_preds = self.generate_predictions(df, 'domain')
        temporal_preds = self.generate_predictions(df, 'temporal')
        
        # Combine predictions
        fusion_inputs = np.column_stack([
            author_preds, domain_preds, temporal_preds
        ])
        
        # Safety check
        fusion_inputs = np.nan_to_num(fusion_inputs, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Make fusion prediction
        fusion_model = self.models['fusion']
        fusion_model.eval()
        
        with torch.no_grad():
            inputs = torch.FloatTensor(fusion_inputs).to(self.device)
            outputs = fusion_model(inputs)
            return outputs.cpu().numpy().flatten()
    
    def plot_learning_curves(self):
        """Plot learning curves for all models"""
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 10))
        axes = axes.flatten()
        
        for i, (model_name, history) in enumerate(self.histories.items()):
            if i < len(axes):
                ax = axes[i]
                ax.plot(history['train_loss'], label='Train')
                ax.plot(history['val_loss'], label='Validation')
                ax.set_title(f'{model_name.capitalize()} Model')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss (Huber)')
                ax.legend()
        
        plt.tight_layout()
        plt.savefig('learning_curves.png')
        plt.show()
    
    def feature_importance(self):
        """Analyze feature importance by training simple models and comparing performance"""
        results = {}
        
        for model_name in ['author', 'domain', 'temporal']:
            if model_name == 'author':
                features = self.author_features
            elif model_name == 'domain':
                features = self.domain_features
            elif model_name == 'temporal':
                features = self.temporal_features
                
            print(f"\nAnalyzing importance of {model_name} features:")
            for feature in features:
                # Train model with just this feature
                X_train = self.train_df[[feature]].values
                y_train = self.train_df[self.target].values
                X_test = self.test_df[[feature]].values
                y_test = self.test_df[self.target].values
                
                # Standardize
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Simple linear model
                model = nn.Linear(1, 1).to(self.device)
                criterion = nn.HuberLoss(delta=1.0)
                optimizer = optim.Adam(model.parameters(), lr=0.0001)
                
                # Train
                model.train()
                for _ in range(100):  # Simple training loop
                    inputs = torch.FloatTensor(X_train_scaled).to(self.device)
                    targets = torch.FloatTensor(y_train.reshape(-1, 1)).to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                
                # Evaluate
                model.eval()
                with torch.no_grad():
                    inputs = torch.FloatTensor(X_test_scaled).to(self.device)
                    outputs = model(inputs)
                    preds = outputs.cpu().numpy().flatten()
                    
                log_mse = np.mean((preds - y_test) ** 2)
                log_rmse = np.sqrt(log_mse)
                
                print(f"  {feature}: Log RMSE = {log_rmse:.4f}")
                results[feature] = log_rmse
                
        return results
    
    def run_pipeline(self):
        """Run the complete training pipeline"""
        print("Preparing data...")
        self.prepare_data()
        
        print("Training individual models...")
        self.train_all_models()
        
        print("Training fusion model...")
        self.train_fusion_model()
        
        print("Evaluating models...")
        results = self.evaluate()
        
        # Print results in log scale (more relevant for model comparison)
        print("\nLog-scale metrics (how models were trained):")
        for model_name, metrics in results.items():
            print(f"{model_name.capitalize()} Model - Log RMSE: {metrics['Log_RMSE']:.4f}, Log MAE: {metrics['Log_MAE']:.4f}")
        
        # Print results in original scale (more interpretable)
        print("\nOriginal-scale metrics (easier to interpret):")
        for model_name, metrics in results.items():
            print(f"{model_name.capitalize()} Model - RMSE: {metrics['RMSE']:.4f}, MAE: {metrics['MAE']:.4f}")
        
        print("\nAnalyzing feature importance...")
        self.feature_importance()
        
        print("\nPlotting learning curves...")
        self.plot_learning_curves()
        
        return results

# Function to prepare categorical features needed by models
def prepare_categorical_features(df):
    """
    Prepare categorical features for model training
    Args:
        df: DataFrame with features
    Returns:
        DataFrame with added encoded categorical features
    """
    # Encode weekday
    weekday_map = {
        'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
        'Friday': 4, 'Saturday': 5, 'Sunday': 6
    }
    if 'weekday' in df.columns:
        df['weekday_encoded'] = df['weekday'].map(weekday_map)
    
    # Fill NaN values for numerical features
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isna().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
    
    return df

# Optional: Alternative approach for fusion using weighted average
def weighted_average_fusion(author_preds, domain_preds, temporal_preds, weights=None):
    """
    Alternative fusion approach using weighted average
    Args:
        author_preds: Author model predictions
        domain_preds: Domain model predictions
        temporal_preds: Temporal model predictions
        weights: List of weights [author_w, domain_w, temporal_w]
    Returns:
        Weighted average predictions
    """
    if weights is None:
        # Default weights - could be tuned via grid search
        weights = [0.4, 0.4, 0.2]
    
    weighted_preds = (
        weights[0] * author_preds + 
        weights[1] * domain_preds + 
        weights[2] * temporal_preds
    )
    
    return weighted_preds

# Usage example
if __name__ == "__main__":
    # Example usage
    import sqlalchemy
    
    # Assuming HNFeatureEngineer exists and works as before
    hn_engineer = HNFeatureEngineer("postgresql://sy91dhb:g5t49ao@178.156.142.230:5432/hd64m1ki")
    
    # Extract and process data
    df = hn_engineer.process_data(limit=100000)  # Using smaller dataset for faster training
    
    # Prepare categorical features
    df = prepare_categorical_features(df)
    
    print(f"Data shape: {df.shape}")
    
    # Create and run the trainer
    trainer = HNLateFusionTrainer(df)
    results = trainer.run_pipeline()
    
    # Print final results
    print("\nFinal Evaluation Results:")
    for model_name, metrics in results.items():
        improvement = (1 - metrics['Log_RMSE'] / results['fusion']['Log_RMSE']) * 100 if model_name != 'fusion' else 0
        print(f"{model_name.capitalize()} Model - Log RMSE: {metrics['Log_RMSE']:.4f}" + 
              (f" ({improvement:.1f}% {'better' if improvement > 0 else 'worse'} than fusion)" if model_name != 'fusion' else ""))