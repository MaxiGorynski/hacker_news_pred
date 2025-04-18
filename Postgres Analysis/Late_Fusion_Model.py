import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import re
from datetime import datetime, timedelta
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import sqlalchemy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from tqdm import tqdm

################### Feature Engineering #########################


#nltk.download('vader_lexicon')
#nltk.download('stopwords')

# Load pkl file as df (Option 1)
#df = pd.read_pickle("hn_10_years_data.pkl")

# Connect with Postgresql database

class HNFeatureEngineer:
    def __init__(self, db_connection_string):
        """Initialize with database connection string"""
        self.db_connection = db_connection_string
        self.engine = sqlalchemy.create_engine(db_connection_string)
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.stopwords = set(nltk.corpus.stopwords.words('english'))

    def extract_base_data(self, start_date=None, end_date=None, limit=None):
        """Extract base data from PostgreSQL"""
        # Define date range for query
        if start_date is None:
            start_date = datetime.now() - timedelta(days=365)
        if end_date is None:
            end_date = datetime.now()

        # Build query
        query = f"""
        SELECT 
            i.id,
            i.title,
            COALESCE(SUBSTRING(i.url FROM 'https?://([^/]+)'), 'unknown') AS domain,
            i.by AS author,
            i.time AS time_of_post, 
            i.score as upvotes,
            u.karma as author_karma,
            u.created as user_created_at,
            i.descendants AS comment_count  
        FROM 
            hacker_news.items i
        JOIN 
            hacker_news.users u ON i.by = u.id
        WHERE 
            i.type = 'story'
            AND i.title IS NOT NULL
            AND i.score IS NOT NULL
            AND i.dead IS DISTINCT FROM TRUE
            AND i.time BETWEEN '{start_date.isoformat()}' AND '{end_date.isoformat()}'
        ORDER BY 
            i.time DESC
        """

        if limit:
            query += f" LIMIT {limit}"

        # Execute query
        df = pd.read_sql(query, self.engine)
        df['time_of_post'] = pd.to_datetime(df['time_of_post'])
        df['year'] = df['time_of_post'].dt.year
        df['month'] = df['time_of_post'].dt.month_name()
        df['period'] = df['year'].astype(str) + '-' + df['time_of_post'].dt.month.apply(lambda m: 'H1' if m <= 6 else 'H2')
        df['weekday'] = df['time_of_post'].dt.day_name()
        df['time_hhmm'] = df['time_of_post'].dt.strftime('%H:%M')
        return df

    def extract_domain_metrics(self, df):
        """Calculate domain-based metrics"""
        # Extract domain statistics
        domain_stats = df.groupby('domain').agg(
            domain_post_count=('upvotes', 'count'),
            domain_avg_upvotes=('upvotes', 'mean'),
            domain_median_upvotes=('upvotes', 'median'),
            domain_max_upvotes=('upvotes', 'max')
        ).reset_index()

        # Calculate domain trend (last 6 months vs previous 6 months)
        now = df['time_of_post'].max()
        three_months_ago = now - timedelta(days=90)
        six_months_ago = now - timedelta(days=180)

        recent_posts = df[df['time_of_post'] > three_months_ago]
        older_posts = df[(df['time_of_post'] <= three_months_ago) & (df['time_of_post'] > six_months_ago)]

        recent_domain_stats = recent_posts.groupby('domain')['upvotes'].mean().reset_index()
        recent_domain_stats.columns = ['domain', 'recent_avg_upvotes']

        older_domain_stats = older_posts.groupby('domain')['upvotes'].mean().reset_index()
        older_domain_stats.columns = ['domain', 'older_avg_upvotes']

        domain_trend = recent_domain_stats.merge(older_domain_stats, on='domain', how='left')
        domain_trend['domain_trend'] = domain_trend['recent_avg_upvotes'] / domain_trend['older_avg_upvotes'].fillna(domain_trend['recent_avg_upvotes'])
        domain_trend = domain_trend[['domain', 'domain_trend']]

        # Merge all domain metrics
        domain_metrics = domain_stats.merge(domain_trend, on='domain', how='left')
        domain_metrics = domain_metrics.sort_values(by='domain_post_count', ascending=False).reset_index(drop=True)
        
        return domain_metrics

    def extract_author_metrics(self, df):
        """Calculate author-based metrics"""
        # Extract author statistics
        author_stats = df.groupby('author').agg(
            author_post_count=('upvotes', 'count'),
            author_avg_upvotes=('upvotes', 'mean'),
            author_median_upvotes=('upvotes', 'median'),
            author_max_upvotes=('upvotes', 'max')
        ).reset_index()

        # Calculate author consistency (std deviation of upvotes)
        author_consistency = df.groupby('author')['upvotes'].std().reset_index()
        author_consistency.columns = ['author', 'author_upvote_std']

        # Calculate author trend (last 6 months vs previous 6 months)
        now = df['time_of_post'].max()
        three_months_ago = now - timedelta(days=90)
        six_months_ago = now - timedelta(days=180)

        recent_posts = df[df['time_of_post'] > three_months_ago]
        previous_posts = df[(df['time_of_post'] <= three_months_ago) & (df['time_of_post'] > six_months_ago)]

        # Group both
        recent_avg = recent_posts.groupby('author')['upvotes'].mean().reset_index()
        recent_avg.columns = ['author', 'recent_avg_upvotes']

        previous_avg = previous_posts.groupby('author')['upvotes'].mean().reset_index()
        previous_avg.columns = ['author', 'previous_avg_upvotes']

        # Merge + calculate trend
        trend = recent_avg.merge(previous_avg, on='author', how='inner')
        trend['author_trend'] = trend['recent_avg_upvotes'] / trend['previous_avg_upvotes'].replace(0, np.nan)
        trend['author_trend'] = trend['author_trend'].fillna(1.0)
        trend = trend[['author', 'author_trend']]
                
        # Final merge: combine everything
        author_metrics = author_stats.merge(author_consistency, on='author', how='left')
        author_metrics = author_metrics.merge(trend, on='author', how='left')

        # Sort by most active authors (optional)
        author_metrics = author_metrics.sort_values(by='author_post_count', ascending=False).reset_index(drop=True)

        return author_metrics


    def extract_subject_matter(self, df, n_topics=10):
        """Extract subject matter from titles using TF-IDF"""
        # Preprocess titles: lowercase and remove non-word characters
        df['clean_title'] = df['title'].str.lower().apply(lambda x: re.sub(r'[^\w\s]', ' ', str(x)))
    
        # TF-IDF vectorization with lower max_features to get more prominent terms
        tfidf = TfidfVectorizer(max_features=n_topics, 
                            stop_words='english',
                            min_df=5,  # Only consider terms that appear in at least 5 documents
                            max_df=0.8)  # Ignore terms that appear in more than 80% of documents
    
        tfidf_matrix = tfidf.fit_transform(df['clean_title'])
    
        # Use actual keywords as column names
        feature_names = tfidf.get_feature_names_out()
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)
        tfidf_df.index = df.index  # Align with original dataframe
    
        # Merge with original data
        result_df = pd.concat([df, tfidf_df], axis=1)
        return result_df, feature_names.tolist()

    def create_time_features(self, df):
        """Create time-based features"""
        # Basic time components
        df['post_hour'] = df['time_of_post'].dt.hour
        df['post_month'] = df['time_of_post'].dt.month
        df['post_dayofmonth'] = df['time_of_post'].dt.day
        df['post_weekend'] = df['weekday'].isin(['Saturday', 'Sunday']).astype(int)
        df['post_US_business_hours'] = ((df['post_hour'] >= 14) & (df['post_hour'] <= 22) &
                                        (~df['post_weekend'])).astype(int)

        # User age at post time (days)
        df['user_age_days'] = (df['time_of_post'] - df['user_created_at']).dt.days

        # Time since account creation buckets
        df['user_age_bucket'] = pd.cut(
            df['user_age_days'],
            bins=[0, 30, 90, 365, 365 * 2, 365 * 5, 365 * 10, np.inf],
            labels=['<1mo', '1-3mo', '3-12mo', '1-2yr', '2-5yr', '5-10yr', '>10yr']
        )

        # Add quarter
        df['post_quarter'] = df['time_of_post'].dt.quarter

        return df

    def create_content_features(self, df):
        """Create features from post content"""
        # Title features
        df['title_length'] = df['title'].str.len()
        df['title_word_count'] = df['title'].str.split().str.len()
        df['has_question'] = df['title'].str.contains(r'\?').astype(int)
        df['has_number'] = df['title'].str.contains(r'\d').astype(int)
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

        # Domain features
        df['is_github'] = df['domain'].str.contains('github.com', case=False).fillna(False).astype(int)
        df['is_blog'] = df['domain'].str.contains('blog|medium.com', case=False).fillna(False).astype(int)
        df['is_video'] = df['domain'].str.contains('youtube.com|vimeo|youtu.be', case=False).fillna(False).astype(int)

        return df

    def process_data(self, start_date=None, end_date=None, limit=None):
        """Full data processing pipeline"""
        print("Extracting base data...")
        df = self.extract_base_data(start_date, end_date, limit)
        print(f"Extracted {len(df)} records.")

        print("Creating time features...")
        df = self.create_time_features(df)

        print("Creating content features...")
        df = self.create_content_features(df)

        print("Extracting domain metrics...")
        domain_metrics = self.extract_domain_metrics(df)
        df = df.merge(domain_metrics, on='domain', how='left')

        print("Extracting author metrics...")
        author_metrics = self.extract_author_metrics(df)
        df = df.merge(author_metrics, on='author', how='left')

        print("Extracting subject matter...")
        df, topic_terms = self.extract_subject_matter(df)
        print(f"Extracted {len(topic_terms)} topic terms: {', '.join(topic_terms[:10])}...")

        print("Feature engineering complete.")
        return df

    def get_feature_names(self):
        """Return list of engineered feature names"""
        # Primary features
        time_features = ['post_hour', 'post_day', 'post_month', 'post_dayofmonth',
                         'post_weekend', 'post_US_business_hours', 'post_quarter']

        user_features = ['author_karma', 'user_age_days', 'user_age_bucket',
                         'author_post_count', 'author_avg_upvotes', 'author_median_upvotes',
                         'author_max_upvotes', 'author_upvote_std', 'author_trend']

        content_features = ['title_length', 'title_word_count', 'has_question', 'has_number',
                            'has_code_reference', 'sentiment_neg', 'sentiment_neu',
                            'sentiment_pos', 'sentiment_compound', 'comment_count',
                            'is_github', 'is_blog', 'is_video']

        domain_features = ['domain_post_count', 'domain_avg_upvotes', 'domain_median_upvotes',
                           'domain_max_upvotes', 'domain_trend']

        # We exclude topic features here as they are dynamically generated
        return {
            'time_features': time_features,
            'user_features': user_features,
            'content_features': content_features,
            'domain_features': domain_features
        }


####################### Late Fusion Model #####################

# Define simplified models for each feature group
class TemporalModel(nn.Module):
    def __init__(self, input_dim=2):  # Change this to accept original input_dim
        super(TemporalModel, self).__init__()
        # Define constants for embedding sizes
        weekday_dim = 7  # 0-6 for days of week
        time_bucket_dim = 4  # 0-3 for time buckets
        embedding_dim = 8
        hidden_dim = 32
        
        # Embeddings for weekday and time buckets
        self.weekday_embedding = nn.Embedding(weekday_dim, embedding_dim)
        self.time_bucket_embedding = nn.Embedding(time_bucket_dim, embedding_dim)
        
        # MLP for processing combined embeddings
        self.regressor = nn.Sequential(
            nn.Linear(2 * embedding_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        # Extract features and ensure they're proper integers in range
        # Clip values to valid ranges to prevent index errors
        time_bucket = torch.clamp(x[:, 0].long(), min=0, max=3)
        weekday = torch.clamp(x[:, 1].long(), min=0, max=6)
        
        # Get embeddings
        time_emb = self.time_bucket_embedding(time_bucket)
        weekday_emb = self.weekday_embedding(weekday)
        
        # Concatenate embeddings
        combined = torch.cat([time_emb, weekday_emb], dim=1)
        
        # Pass through regressor
        return self.regressor(combined)

class TitleLengthModel(nn.Module):
    def __init__(self, input_dim):
        super(TitleLengthModel, self).__init__()
        self.regressor = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.5),
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
            nn.Dropout(0.2),
            nn.Linear(8, 4),
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
        df = prepare_temporal_features(df)
        self.df = df
        self.device = device
        self.scalers = {}
        self.models = {}
        self.histories = {}
        
        # Feature sets
        self.temporal_features = ["time_bucket_encoded", "weekday_encoded"]
        self.title_features = ["title_length"]
        
        # Ensure features exist
        all_features = set(df.columns)
        self.temporal_features = [f for f in self.temporal_features if f in all_features]
        self.title_features = [f for f in self.title_features if f in all_features]

        # Target variable
        print("Applying log transform to target variable")
        self.df['upvotes_log'] = np.log1p(self.df['upvotes'])
        self.target = 'upvotes_log'
        
        # No need to preprocess author/domain stats
        print("Initialized with temporal and title-length features")
        
 
    def validate_data(self):
        """Check data for NaN, infinity, or extreme values"""
        for feature_group, features in [
            ('temporal', self.temporal_features),
            ('title', self.title_features)
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
        print("Preparing temporal dataset...")
        temporal_train_loader, temporal_val_loader = self._prepare_feature_data(
            train_df, test_df, self.temporal_features, 'temporal'
        )

        print("Preparing title-length dataset...")
        title_train_loader, title_val_loader = self._prepare_feature_data(
            train_df, test_df, self.title_features, 'title'
        )

        self.loaders = {
            'temporal': (temporal_train_loader, temporal_val_loader),
            'title': (title_train_loader, title_val_loader),
        }
        
        # Store data for fusion model training later
        self.train_df = train_df
        self.test_df = test_df
    
    def _prepare_feature_data(self, train_df, test_df, feature_list, name):
        """Prepare dataset for a feature group with robust preprocessing"""
        try:
            X_train = train_df[feature_list].copy()
            y_train = train_df[self.target].values

            X_test = test_df[feature_list].copy()
            y_test = test_df[self.target].values

            X_train = X_train.replace([np.inf, -np.inf], np.nan)
            X_test = X_test.replace([np.inf, -np.inf], np.nan)

            for col in X_train.columns:
                median_val = X_train[col].median()
                X_train[col] = X_train[col].fillna(median_val)
                X_test[col] = X_test[col].fillna(median_val)

            for col in X_train.columns:
                q5 = X_train[col].quantile(0.05)
                q95 = X_train[col].quantile(0.95)
                X_train[col] = X_train[col].clip(q5, q95)
                X_test[col] = X_test[col].clip(q5, q95)

            if name == 'temporal':
                X_train_scaled = X_train.values
                X_test_scaled = X_test.values
                self.scalers[name] = None
            else:
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                self.scalers[name] = scaler

            X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0, posinf=0.0, neginf=0.0)
            X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0, posinf=0.0, neginf=0.0)

            train_data = TensorDataset(
                torch.FloatTensor(X_train_scaled),
                torch.FloatTensor(y_train.reshape(-1, 1))
            )
            test_data = TensorDataset(
                torch.FloatTensor(X_test_scaled),
                torch.FloatTensor(y_test.reshape(-1, 1))
            )

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

        criterion = nn.HuberLoss(delta=1.0)
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

        best_val_loss = float('inf')
        patience_counter = 0
        history = {'train_loss': [], 'val_loss': []}

        for epoch in range(num_epochs):
            model.train()
            train_losses = []

            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                if torch.isnan(inputs).any() or torch.isinf(inputs).any():
                    print(f"Warning: NaN or Inf detected in inputs for {model_name}")
                    inputs = torch.nan_to_num(inputs)

                optimizer.zero_grad()
                outputs = model(inputs)

                if torch.isnan(outputs).any():
                    print("NaN detected in outputs, skipping batch")
                    continue

                loss = criterion(outputs, targets)

                if torch.isnan(loss).any():
                    print("NaN detected in loss, skipping batch")
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()

                train_losses.append(loss.item())

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

            if np.isnan(train_loss) or np.isinf(train_loss):
                print(f"Warning: Train loss is {train_loss} at epoch {epoch+1}")
            if np.isnan(val_loss) or np.isinf(val_loss):
                print(f"Warning: Val loss is {val_loss} at epoch {epoch+1}")

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)

            print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
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
        # Train temporal model
        temporal_model = TemporalModel(input_dim=len(self.temporal_features))
        temporal_train_loader, temporal_val_loader = self.loaders['temporal']
        self.train_model('temporal', temporal_model, temporal_train_loader, temporal_val_loader)

        # Train title model
        title_length_model = TitleLengthModel(input_dim=len(self.title_features))
        title_train_loader, title_val_loader = self.loaders['title']
        self.train_model('title', title_length_model, title_train_loader, title_val_loader)
    
    def generate_predictions(self, df, model_name):
        """Generate predictions for a specific model"""
        model = self.models[model_name]
        model.eval()
        
        with torch.no_grad():
            # Get features for the model
            if model_name == 'temporal':
                features = self.temporal_features
            elif model_name == 'title':
                features = self.title_features
            else:
                raise ValueError(f"Unsupported model_name: {model_name}")
            
            X = df[features].values
            
            if self.scalers.get(model_name) is None:
                X_scaled = X  # Keep raw ints for embedding
            else:
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
        train_temporal_preds = self.generate_predictions(self.train_df, 'temporal')
        train_title_preds = self.generate_predictions(self.train_df, 'title')
        
        # Generate predictions for test data
        test_temporal_preds = self.generate_predictions(self.test_df, 'temporal')
        test_title_preds = self.generate_predictions(self.test_df, 'title')

        # Combine predictions
        train_fusion_inputs = np.column_stack([
            train_temporal_preds, train_title_preds
        ])

        test_fusion_inputs = np.column_stack([
            test_temporal_preds, test_title_preds
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
        fusion_model = FusionModel(num_models=2)
        self.train_model('fusion', fusion_model, train_fusion_loader, test_fusion_loader)
    
    def evaluate(self):
        """Evaluate all models on test data"""
        results = {}
        
        # Individual models evaluation
        for model_name in ['temporal', 'title']:
            preds = self.generate_predictions(self.test_df, model_name)
            
            # First evaluate on log scale (what model was trained on)
            log_true_values = self.test_df[self.target].values
            log_mse = np.mean((preds - log_true_values) ** 2)
            log_rmse = np.sqrt(log_mse)
            log_mae = np.mean(np.abs(preds - log_true_values))
            
            # Convert back to original scale for interpretability
            preds_clipped = np.clip(preds, -10, 10)
            preds_original = np.expm1(preds_clipped)
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
        fusion_preds_clipped = np.clip(fusion_preds, -10, 10)
        fusion_preds_original = np.expm1(fusion_preds_clipped)
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
        temporal_preds = self.generate_predictions(df, 'temporal')
        title_preds = self.generate_predictions(df, 'title')
        
        # Combine predictions
        fusion_inputs = np.column_stack([
            temporal_preds, title_preds
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
        num_models = len(self.histories)
        fig, axes = plt.subplots(nrows=1, ncols=num_models, figsize=(6 * num_models, 4))

        if num_models == 1:
            axes = [axes]  # Make iterable if only one plot

        for ax, (model_name, history) in zip(axes, self.histories.items()):
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

        # Define the only relevant feature groups
        feature_sets = {
            'temporal': self.temporal_features,
            'title': ['title_length']  # assuming you use this feature
        }
        
        for model_name, features in feature_sets.items():
            print(f"\nAnalyzing importance of {model_name} features:")
            for feature in features:
                X_train = self.train_df[[feature]].values
                y_train = self.train_df[self.target].values
                X_test = self.test_df[[feature]].values
                y_test = self.test_df[self.target].values

                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

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
def prepare_temporal_features(df):
    """
    Prepare temporal features with time buckets for model training
    """
    # Define time bucket function
    def time_bucket(hour):
        hour = int(hour)  # Ensure hour is an integer
        if 5 <= hour < 11:
            return 0  # Morning
        elif 11 <= hour < 17:
            return 1  # Afternoon
        elif 17 <= hour < 21:
            return 2  # Evening
        else:
            return 3  # Night
    
    # Create time bucket feature directly as integers (0-3)
    df['time_bucket_encoded'] = df['post_hour'].apply(time_bucket)
    
    # Encode weekday if needed
    if 'weekday' in df.columns and 'weekday_encoded' not in df.columns:
        weekday_map = {
            'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
            'Friday': 4, 'Saturday': 5, 'Sunday': 6
        }
        df['weekday_encoded'] = df['weekday'].map(weekday_map)
    
    # Ensure values are in proper range
    df['time_bucket_encoded'] = df['time_bucket_encoded'].clip(0, 3)
    df['weekday_encoded'] = df['weekday_encoded'].clip(0, 6)
    
    return df

# Optional: Alternative approach for fusion using weighted average
def weighted_average_fusion(temporal_preds, title_preds, weights=None):
    """
    Alternative fusion approach using weighted average
    Args:
        temporal_preds: Temporal model predictions
        title_preds: Title model predictions
        weights: List of weights [temporal_w, title_w]
    Returns:
        Weighted average predictions
    """
    if weights is None:
        # Default weights can be tuned further
        weights = [0.5, 0.5]

    return weights[0] * temporal_preds + weights[1] * title_preds

def add_title_length_feature(df):
    """
    Adds a feature representing the length of the title (in tokens).
    """
    df['title_length'] = df['title'].fillna("").apply(lambda t: len(str(t).split()))
    return df

# Usage example
if __name__ == "__main__":
    # Example usage
    import sqlalchemy
    
    # Assuming HNFeatureEngineer exists and works as before
    hn_engineer = HNFeatureEngineer("postgresql://sy91dhb:g5t49ao@178.156.142.230:5432/hd64m1ki")
    
    # Extract and process data
    df = hn_engineer.process_data(limit=100000)  # Using smaller dataset for faster training
    
    # Apply temporal feature preprocessing & tittle length calculation
    df = prepare_temporal_features(df)
    df = add_title_length_feature(df)
    
    print(f"Data shape: {df.shape}")
    
    # Create and run the trainer
    trainer = HNLateFusionTrainer(df)
    results = trainer.run_pipeline()
    
    # Print final results
    print("\nFinal Evaluation Results:")
    for model_name in ['temporal', 'title', 'fusion']:
        metrics = results[model_name]
        improvement = (1 - metrics['Log_RMSE'] / results['fusion']['Log_RMSE']) * 100 if model_name != 'fusion' else 0
        print(f"{model_name.capitalize()} Model - Log RMSE: {metrics['Log_RMSE']:.4f}" + 
              (f" ({improvement:.1f}% {'better' if improvement > 0 else 'worse'} than fusion)" if model_name != 'fusion' else ""))