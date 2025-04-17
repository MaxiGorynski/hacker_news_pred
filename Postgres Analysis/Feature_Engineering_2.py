import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import re
from datetime import datetime, timedelta
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import sqlalchemy

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
            start_date = datetime.now() - timedelta(days=10*365)
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
        six_months_ago = now - timedelta(days=180)
        twelve_months_ago = now - timedelta(days=360)

        recent_posts = df[df['time_of_post'] > six_months_ago]
        older_posts = df[(df['time_of_post'] <= six_months_ago) & (df['time_of_post'] > twelve_months_ago)]

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
        six_months_ago = now - timedelta(days=180)
        twelve_months_ago = now - timedelta(days=360)

        recent_posts = df[df['time_of_post'] > six_months_ago]
        previous_posts = df[(df['time_of_post'] <= six_months_ago) & (df['time_of_post'] > twelve_months_ago)]

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