import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb
import xgboost as xgb
import shap
import joblib
from datetime import datetime, timedelta
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('HNModelTrainer')


class HNModelTrainer:
    def __init__(self, feature_data_path=None, feature_data_df=None):
        """Initialize with either a path to feature data or a DataFrame"""
        if feature_data_df is not None:
            self.data = feature_data_df
        elif feature_data_path is not None:
            self.data = pd.read_csv(feature_data_path)
            # Convert date columns from string to datetime
            date_columns = [col for col in self.data.columns if 'date' in col.lower() or 'created_at' in col.lower()]
            for col in date_columns:
                if col in self.data.columns:
                    self.data[col] = pd.to_datetime(self.data[col])
        else:
            raise ValueError("Either feature_data_path or feature_data_df must be provided")

        self.models = {}
        self.feature_importances = {}
        self.preprocessor = None
        self.target_transformer = None
        self.evaluation_results = {}

    def prepare_data(self, target_col='upvotes', test_size=0.2, random_state=42, time_based_split=True):
        """Prepare data for model training"""
        logger.info("Preparing data for modeling...")

        # Define feature groups
        numeric_features = self.data.select_dtypes(include=['int64', 'float64']).columns.tolist()
        numeric_features = [f for f in numeric_features if f != target_col and not f.startswith('topic_')]

        categorical_features = self.data.select_dtypes(include=['object']).columns.tolist()
        categorical_features = [f for f in categorical_features if f != 'id' and f != 'title' and f != 'url'
                                and f != 'clean_title' and 'created_at' not in f]

        # Add any topic features
        topic_features = [col for col in self.data.columns if col.startswith('topic_')]

        # Log features
        logger.info(f"Using {len(numeric_features)} numeric features")
        logger.info(f"Using {len(categorical_features)} categorical features")
        logger.info(f"Using {len(topic_features)} topic features")

        # Create preprocessor
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='passthrough'  # This will include the topic features
        )

        self.preprocessor = preprocessor
        self.feature_names = numeric_features + categorical_features + topic_features

        # Log transformation of target variable
        logger.info(f"Target variable: {target_col}")

        # Define target variable
        X = self.data.drop(columns=[target_col, 'id', 'title', 'url', 'clean_title'])
        if 'created_at' in X.columns:
            X = X.drop(columns=['created_at'])
        if 'user_created_at' in X.columns:
            X = X.drop(columns=['user_created_at'])

        y = self.data[target_col]

        # Split data
        if time_based_split and 'timestamp' in self.data.columns:
            # Sort by timestamp
            sorted_indices = self.data['timestamp'].argsort()
            X = X.iloc[sorted_indices]
            y = y.iloc[sorted_indices]

            # Use the last test_size portion as test set
            split_idx = int(len(X) * (1 - test_size))
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

            logger.info(
                f"Using time-based split: train from {X.index[0]} to {X.index[split_idx - 1]}, test from {X.index[split_idx]} to {X.index[-1]}")
        else:
            # Random split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            logger.info(f"Using random split: {len(X_train)} training samples, {len(X_test)} test samples")

        return X_train, X_test, y_train, y_test

    def train_random_forest(self, X_train, y_train, param_grid=None):
        """Train a Random Forest model"""
        logger.info("Training Random Forest model...")

        if param_grid is None:
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }

        # Create and train model
        model = RandomForestRegressor(random_state=42)

        # Random search for hyperparameters
        random_search = RandomizedSearchCV(
            model, param_distributions=param_grid,
            n_iter=10, cv=3, random_state=42, n_jobs=-1,
            scoring='neg_mean_squared_error'
        )

        # Fit model
        random_search.fit(X_train, y_train)

        # Get best model
        best_model = random_search.best_estimator_
        logger.info(f"Best Random Forest parameters: {random_search.best_params_}")

        # Store model
        self.models['random_forest'] = best_model

        # Extract feature importances
        feature_importances = best_model.feature_importances_
        self.feature_importances['xgboost'] = feature_importances

        return best_model

    def train_lightgbm(self, X_train, y_train, param_grid=None):
        """Train a LightGBM model"""
        logger.info("Training LightGBM model...")

        if param_grid is None:
            param_grid = {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 5, 7, -1],
                'num_leaves': [31, 50, 100],
                'min_child_samples': [20, 30, 50],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0],
                'reg_alpha': [0, 0.1, 0.5],
                'reg_lambda': [0, 0.1, 0.5]
            }

        # Create and train model
        model = lgb.LGBMRegressor(random_state=42)

        # Random search for hyperparameters
        random_search = RandomizedSearchCV(
            model, param_distributions=param_grid,
            n_iter=10, cv=3, random_state=42, n_jobs=-1,
            scoring='neg_mean_squared_error'
        )

        # Fit model
        random_search.fit(X_train, y_train)

        # Get best model
        best_model = random_search.best_estimator_
        logger.info(f"Best LightGBM parameters: {random_search.best_params_}")

        # Store model
        self.models['lightgbm'] = best_model

        # Extract feature importances
        feature_importances = best_model.feature_importances_
        self.feature_importances['lightgbm'] = feature_importances

        return best_model

    def evaluate_model(self, model, X_test, y_test, model_name=None):
        """Evaluate model performance"""
        if model_name is None:
            model_name = type(model).__name__

        logger.info(f"Evaluating {model_name} model...")

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        # Store evaluation results
        self.evaluation_results[model_name] = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'predictions': y_pred,
            'actuals': y_test
        }

        logger.info(f"Model {model_name} evaluation results:")
        logger.info(f"  MAE: {mae:.2f}")
        logger.info(f"  RMSE: {rmse:.2f}")
        logger.info(f"  R²: {r2:.4f}")

        return mae, rmse, r2

    def plot_feature_importance(self, model_name, top_n=20):
        """Plot feature importance for a model"""
        if model_name not in self.feature_importances:
            logger.error(f"No feature importances found for {model_name}")
            return

        feature_importances = self.feature_importances[model_name]

        # Create DataFrame of feature importances
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': feature_importances
        })

        # Sort by importance
        importance_df = importance_df.sort_values('Importance', ascending=False).head(top_n)

        # Plot
        plt.figure(figsize=(10, 8))
        sns.barplot(x='Importance', y='Feature', data=importance_df)
        plt.title(f'Top {top_n} Feature Importances - {model_name}')
        plt.tight_layout()

        return importance_df

    def shap_analysis(self, model_name, X_sample):
        """Perform SHAP analysis for a model"""
        if model_name not in self.models:
            logger.error(f"Model {model_name} not found")
            return

        model = self.models[model_name]

        # Create explainer
        if model_name == 'random_forest':
            explainer = shap.TreeExplainer(model)
        elif model_name == 'gradient_boosting':
            explainer = shap.TreeExplainer(model)
        elif model_name == 'xgboost':
            explainer = shap.TreeExplainer(model)
        elif model_name == 'lightgbm':
            explainer = shap.TreeExplainer(model)
        else:
            logger.error(f"SHAP analysis not implemented for {model_name}")
            return

        # Calculate SHAP values
        shap_values = explainer.shap_values(X_sample)

        # Plot summary
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample, feature_names=self.feature_names)

        return shap_values

    def save_model(self, model_name, filepath):
        """Save model to disk"""
        if model_name not in self.models:
            logger.error(f"Model {model_name} not found")
            return

        model = self.models[model_name]

        # Save model
        joblib.dump(model, filepath)
        logger.info(f"Model {model_name} saved to {filepath}")

    def compare_models(self):
        """Compare all trained models"""
        if not self.evaluation_results:
            logger.error("No evaluation results found")
            return

        # Create comparison DataFrame
        comparison = pd.DataFrame({
            'Model': [],
            'MAE': [],
            'RMSE': [],
            'R²': []
        })

        for model_name, results in self.evaluation_results.items():
            comparison = pd.concat([comparison, pd.DataFrame({
                'Model': [model_name],
                'MAE': [results['mae']],
                'RMSE': [results['rmse']],
                'R²': [results['r2']]
            })], ignore_index=True)

        # Sort by RMSE (lower is better)
        comparison = comparison.sort_values('RMSE')

        # Plot comparison
        plt.figure(figsize=(12, 6))

        # Plot MAE
        plt.subplot(1, 3, 1)
        sns.barplot(x='Model', y='MAE', data=comparison)
        plt.title('MAE by Model')
        plt.xticks(rotation=45)

        # Plot RMSE
        plt.subplot(1, 3, 2)
        sns.barplot(x='Model', y='RMSE', data=comparison)
        plt.title('RMSE by Model')
        plt.xticks(rotation=45)

        # Plot R²
        plt.subplot(1, 3, 3)
        sns.barplot(x='Model', y='R²', data=comparison)
        plt.title('R² by Model')
        plt.xticks(rotation=45)

        plt.tight_layout()

        return comparison

    def run_full_training(self, target_col='upvotes', test_size=0.2, random_state=42, time_based_split=True):
        """Run full training pipeline"""
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data(
            target_col=target_col,
            test_size=test_size,
            random_state=random_state,
            time_based_split=time_based_split
        )

        # Train models
        self.train_random_forest(X_train, y_train)
        self.train_gradient_boosting(X_train, y_train)
        self.train_xgboost(X_train, y_train)
        self.train_lightgbm(X_train, y_train)

        # Evaluate models
        for name, model in self.models.items():
            self.evaluate_model(model, X_test, y_test, name)

        # Compare models
        comparison = self.compare_models()

        # Return best model
        best_model_name = comparison.iloc[0]['Model']
        logger.info(f"Best model: {best_model_name}")

        return self.models[best_model_name]


self.feature_importances['random_forest'] = feature_importances

return best_model


def train_gradient_boosting(self, X_train, y_train, param_grid=None):
    """Train a Gradient Boosting model"""
    logger.info("Training Gradient Boosting model...")

    if param_grid is None:
        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'subsample': [0.8, 0.9, 1.0]
        }

    # Create and train model
    model = GradientBoostingRegressor(random_state=42)

    # Random search for hyperparameters
    random_search = RandomizedSearchCV(
        model, param_distributions=param_grid,
        n_iter=10, cv=3, random_state=42, n_jobs=-1,
        scoring='neg_mean_squared_error'
    )

    # Fit model
    random_search.fit(X_train, y_train)

    # Get best model
    best_model = random_search.best_estimator_
    logger.info(f"Best Gradient Boosting parameters: {random_search.best_params_}")

    # Store model
    self.models['gradient_boosting'] = best_model

    # Extract feature importances
    feature_importances = best_model.feature_importances_
    self.feature_importances['gradient_boosting'] = feature_importances

    return best_model


def train_xgboost(self, X_train, y_train, param_grid=None):
    """Train an XGBoost model"""
    logger.info("Training XGBoost model...")

    if param_grid is None:
        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'min_child_weight': [1, 3, 5],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'gamma': [0, 0.1, 0.2]
        }

    # Create and train model
    model = xgb.XGBRegressor(random_state=42)

    # Random search for hyperparameters
    random_search = RandomizedSearchCV(
        model, param_distributions=param_grid,
        n_iter=10, cv=3, random_state=42, n_jobs=-1,
        scoring='neg_mean_squared_error'
    )

    # Fit model
    random_search.fit(X_train, y_train)

    # Get best model
    best_model = random_search.best_estimator_
    logger.info(f"Best XGBoost parameters: {random_search.best_params_}")

    # Store model
    self.models['xgboost'] = best_model

    # Extract feature importances
    feature_importances = best_model.feature_importances_