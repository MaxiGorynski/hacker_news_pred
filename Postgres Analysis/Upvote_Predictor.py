import pandas as pd
import numpy as np
import joblib
import sqlalchemy
from datetime import datetime
import logging
import os
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import re

# Download NLTK resources
nltk.download('vader_lexicon', quiet=True)
nltk.download('stopwords', quiet=True)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('HNPredictor')


class HNPredictor:
    def __init__(self, model_path, feature_engineer=None, db_connection_string=None):
        """
        Initialize the predictor

        Parameters:
        -----------
        model_path : str
            Path to the trained model file
        feature_engineer : object, optional
            Feature engineering object with process_data method
        db_connection_string : str, optional
            Database connection string for fetching additional data
        """
        self.model = joblib.load(model_path)
        self.feature_engineer = feature_engineer
        self.db_connection_string = db_connection_string

        if db_connection_string:
            self.engine = sqlalchemy.create_engine(db_connection_string)

        # Initialize sentiment analyzer
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.stopwords = set(nltk.corpus.stopwords.words('english'))

        logger.info(f"Loaded model from {model_path}")

    def fetch_reference_data(self):
        """Fetch reference data from database for feature engineering"""
        if not self.db_connection_string:
            logger.warning("No database connection string provided, skipping reference data fetch")
            return {}

        try:
            # Fetch domain statistics
            domain_query = """
            SELECT 
                COALESCE(domain, 'unknown') as domain,
                COUNT(*) as domain_post_count,
                AVG(score) as domain_avg_upvotes,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY score) as domain_median_upvotes,
                MAX(score) as domain_max_upvotes
            FROM 
                posts
            WHERE 
                created_at > NOW() - INTERVAL '1 year'
                AND score IS NOT NULL
            GROUP BY 
                domain
            """
            domain_stats = pd.read_sql(domain_query, self.engine)

            # Fetch author statistics
            author_query = """
            SELECT 
                author,
                COUNT(*) as author_post_count,
                AVG(score) as author_avg_upvotes,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY score) as author_median_upvotes,
                MAX(score) as author_max_upvotes
            FROM 
                posts
            WHERE 
                created_at > NOW() - INTERVAL '1 year'
                AND score IS NOT NULL
            GROUP BY 
                author
            """
            author_stats = pd.read_sql(author_query, self.engine)

            return {
                'domain_stats': domain_stats,
                'author_stats': author_stats
            }

        except Exception as e:
            logger.error(f"Error fetching reference data: {e}")
            return {}

    def engineer_features(self, post_data, reference_data=None):
        """
        Engineer features for a single post or batch of posts

        Parameters:
        -----------
        post_data : DataFrame or dict
            Post data to engineer features for
        reference_data : dict, optional
            Reference data for feature engineering

        Returns:
        --------
        DataFrame
            Engineered features
        """
        # If post_data is a dict, convert to DataFrame
        if isinstance(post_data, dict):
            post_data = pd.DataFrame([post_data])

        # Make a copy to avoid modifying the original
        df = post_data.copy()

        # If we have a feature engineer, use it
        if self.feature_engineer:
            return self.feature_engineer.process_data(df)

        # Otherwise, perform basic feature engineering

        # Process created_at
        if 'created_at' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['created_at']):
            df['created_at'] = pd.to_datetime(df['created_at'])

        # Process user_created_at
        if 'user_created_at' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['user_created_at']):
            df['user_created_at'] = pd.to_datetime(df['user_created_at'])

        # Time features
        if 'created_at' in df.columns:
            df['post_hour'] = df['created_at'].dt.hour
            df['post_day'] = df['created_at'].dt.dayofweek  # 0=Monday, 6=Sunday
            df['post_month'] = df['created_at'].dt.month
            df['post_dayofmonth'] = df['created_at'].dt.day
            df['post_weekend'] = df['post_day'].isin([5, 6]).astype(int)
            df['post_us_business_hours'] = ((df['post_hour'] >= 14) & (df['post_hour'] <= 22) &
                                            (~df['post_weekend'])).astype(int)

        # User age features
        if 'created_at' in df.columns and 'user_created_at' in df.columns:
            df['user_age_days'] = (df['created_at'] - df['user_created_at']).dt.days

            df['user_age_bucket'] = pd.cut(
                df['user_age_days'],
                bins=[0, 30, 90, 365, 365 * 2, 365 * 5, 365 * 10, np.inf],
                labels=['<1mo', '1-3mo', '3-12mo', '1-2yr', '2-5yr', '5-10yr', '>10yr']
            )

        # Title features
        if 'title' in df.columns:
            df['title_length'] = df['title'].str.len()
            df['title_word_count'] = df['title'].str.split().str.len()
            df['has_question'] = df['title'].str.contains('\?').astype(int)
            df['has_number'] = df['title'].str.contains('\d').astype(int)
            df['has_code_reference'] = df['title'].str.contains('|'.join([
                'code', 'program', 'script', 'function', 'API', 'backend', 'frontend',
                'database', 'server', 'client', 'framework', 'library'
            ]), case=False).astype(int)

            # Sentiment analysis
            df['sentiment_scores'] = df['title'].apply(lambda x: self.sentiment_analyzer.polarity_scores(x))
            df['sentiment_neg'] = df['sentiment_scores'].apply(lambda x: x['neg'])
            df['sentiment_neu'] = df['sentiment_scores'].apply(lambda x: x['neu'])
            df['sentiment_pos'] = df['sentiment_scores'].apply(lambda x: x['pos'])
            df['sentiment_compound'] = df['sentiment_scores'].apply(lambda x: x['compound'])
            df.drop('sentiment_scores', axis=1, inplace=True)

            # Clean title for potential topic modeling
            df['clean_title'] = df['title'].str.lower()
            df['clean_title'] = df['clean_title'].apply(lambda x: ' '.join([word for word in re.split(r'\W+', x)
                                                                            if word and word not in self.stopwords]))

        # URL features
        if 'url' in df.columns:
            df['is_github'] = df['url'].str.contains('github.com', case=False).fillna(False).astype(int)
            df['is_blog'] = df['url'].str.contains('blog|medium.com', case=False).fillna(False).astype(int)
            df['is_video'] = df['url'].str.contains('youtube.com|vimeo|youtu.be', case=False).fillna(False).astype(int)

        # Merge reference data
        if reference_data:
            # Merge domain stats
            if 'domain_stats' in reference_data and 'domain' in df.columns:
                df = df.merge(reference_data['domain_stats'], on='domain', how='left')

            # Merge author stats
            if 'author_stats' in reference_data and 'author' in df.columns:
                df = df.merge(reference_data['author_stats'], on='author', how='left')

        # Fill missing values
        df = df.fillna(0)

        return df

    def predict(self, post_data):
        """
        Predict upvotes for a single post or batch of posts

        Parameters:
        -----------
        post_data : DataFrame or dict
            Post data to predict upvotes for

        Returns:
        --------
        np.array
            Predicted upvote counts
        """
        # Fetch reference data
        reference_data = self.fetch_reference_data()

        # Engineer features
        X = self.engineer_features(post_data, reference_data)

        # Remove target column if present
        if 'upvotes' in X.columns:
            X = X.drop(columns=['upvotes'])

        # Remove ID columns
        for col in ['id', 'title', 'url', 'clean_title']:
            if col in X.columns:
                X = X.drop(columns=[col])

        # Remove date columns
        for col in X.columns:
            if 'created_at' in col:
                X = X.drop(columns=[col])

        # Make predictions
        predictions = self.model.predict(X)

        return predictions

    def predict_with_explanation(self, post_data, num_features=10):
        """
        Predict upvotes with feature importance explanation

        Parameters:
        -----------
        post_data : DataFrame or dict
            Post data to predict upvotes for
        num_features : int, optional
            Number of top features to include in explanation

        Returns:
        --------
        dict
            Prediction results with explanation
        """
        import shap

        # Fetch reference data
        reference_data = self.fetch_reference_data()

        # Engineer features
        X = self.engineer_features(post_data, reference_data)

        # Save original data
        original_data = X.copy()

        # Remove target column if present
        if 'upvotes' in X.columns:
            X = X.drop(columns=['upvotes'])

        # Remove ID columns
        for col in ['id', 'title', 'url', 'clean_title']:
            if col in X.columns:
                X = X.drop(columns=[col])

        # Remove date columns
        for col in X.columns:
            if 'created_at' in col:
                X = X.drop(columns=[col])

        # Make predictions
        predictions = self.model.predict(X)

        # Create explainer
        explainer = shap.TreeExplainer(self.model)

        # Calculate SHAP values
        shap_values = explainer.shap_values(X)

        # Get feature importances
        feature_importance = np.abs(shap_values).mean(0)
        feature_importance_dict = dict(zip(X.columns, feature_importance))

        # Sort by importance
        sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
        top_features = sorted_features[:num_features]

        # Create results
        results = []
        for i, pred in enumerate(predictions):
            post_result = {
                'predicted_upvotes': pred,
                'top_influencing_features': {}
            }

            # Add original data
            if isinstance(post_data, dict):
                for key, value in post_data.items():
                    post_result[key] = value
            else:
                for col in post_data.columns:
                    if col in original_data.columns:
                        post_result[col] = original_data.iloc[i][col]

            # Add top features
            for feature, _ in top_features:
                if feature in X.columns:
                    feature_value = X.iloc[i][feature]
                    feature_impact = shap_values[i][X.columns.get_loc(feature)]
                    post_result['top_influencing_features'][feature] = {
                        'value': feature_value,
                        'impact': feature_impact
                    }

            results.append(post_result)

        # Return first result if only one post
        if len(results) == 1:
            return results[0]
        else:
            return results

    def batch_predict(self, posts_df):
        """
        Predict upvotes for a batch of posts

        Parameters:
        -----------
        posts_df : DataFrame
            DataFrame containing post data

        Returns:
        --------
        DataFrame
            Original DataFrame with predictions added
        """
        # Make a copy
        result_df = posts_df.copy()

        # Make predictions
        predictions = self.predict(posts_df)

        # Add predictions to DataFrame
        result_df['predicted_upvotes'] = predictions

        return result_df

    def get_most_important_features(self, num_features=20):
        """
        Get the most important features for prediction

        Parameters:
        -----------
        num_features : int, optional
            Number of features to return

        Returns:
        --------
        dict
            Dictionary of feature importance scores
        """
        # Check if model has feature_importances_ attribute
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_

            # Get feature names
            if hasattr(self.model, 'feature_names_in_'):
                feature_names = self.model.feature_names_in_
            else:
                # Create generic feature names
                feature_names = [f"feature_{i}" for i in range(len(importances))]

            # Create dictionary of feature importances
            importance_dict = dict(zip(feature_names, importances))

            # Sort by importance
            sorted_importances = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)

            # Return top N features
            return dict(sorted_importances[:num_features])
        else:
            logger.warning("Model doesn't have feature_importances_ attribute")
            return {}


# Example usage
if __name__ == "__main__":
    # Initialize predictor
    model_path = "models/hn_upvote_model.joblib"
    db_connection = "postgresql://username:password@localhost:5432/hn_database"

    predictor = HNPredictor(model_path, db_connection_string=db_connection)

    # Example post
    example_post = {
        'title': 'Introducing a new framework for distributed systems',
        'url': 'https://github.com/username/new-framework',
        'domain': 'github.com',
        'author': 'user123',
        'created_at': datetime.now(),
        'user_created_at': datetime.now() - timedelta(days=365 * 2)
    }

    # Predict upvotes
    prediction = predictor.predict(example_post)
    print(f"Predicted upvotes: {prediction[0]:.2f}")

    # Predict with explanation
    explanation = predictor.predict_with_explanation(example_post)
    print("\nPrediction explanation:")
    for feature, details in explanation['top_influencing_features'].items():
        print(f"  {feature}: {details['value']} (impact: {details['impact']:.4f})")

    # Show most important features overall
    important_features = predictor.get_most_important_features()
    print("\nMost important features overall:")
    for feature, importance in important_features.items():
        print(f"  {feature}: {importance:.4f}")