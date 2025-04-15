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

# Download NLTK resources
nltk.download('vader_lexicon')
nltk.download('stopwords')


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
            posts.id,
            posts.title,
            posts.url,
            COALESCE(posts.domain, 'unknown') as domain,
            posts.author,
            posts.created_at,
            posts.score as upvotes,
            users.karma as author_karma,
            users.created_at as user_created_at,
            COUNT(comments.id) as comment_count
        FROM 
            posts
        LEFT JOIN 
            users ON posts.author = users.username
        LEFT JOIN
            comments ON posts.id = comments.post_id
        WHERE 
            posts.created_at BETWEEN '{start_date.isoformat()}' AND '{end_date.isoformat()}'
            AND posts.score IS NOT NULL
        GROUP BY
            posts.id, posts.title, posts.url, posts.domain, posts.author, 
            posts.created_at, posts.score, users.karma, users.created_at
        ORDER BY 
            posts.created_at DESC
        """

        if limit:
            query += f" LIMIT {limit}"

        # Execute query
        df = pd.read_sql(query, self.engine)
        return df

    def extract_domain_metrics(self, df):
        """Calculate domain-based metrics"""
        # Extract domain statistics
        domain_stats = df.groupby('domain')['upvotes'].agg({
            'domain_post_count': 'count',
            'domain_avg_upvotes': 'mean',
            'domain_median_upvotes': 'median',
            'domain_max_upvotes': 'max'
        }).reset_index()

        # Calculate domain trend (last 3 months vs previous 3 months)
        now = datetime.now()
        three_months_ago = now - timedelta(days=90)
        six_months_ago = now - timedelta(days=180)

        recent_posts = df[df['created_at'] > three_months_ago]
        older_posts = df[(df['created_at'] <= three_months_ago) & (df['created_at'] > six_months_ago)]

        recent_domain_stats = recent_posts.groupby('domain')['upvotes'].mean().reset_index()
        recent_domain_stats.columns = ['domain', 'recent_avg_upvotes']

        older_domain_stats = older_posts.groupby('domain')['upvotes'].mean().reset_index()
        older_domain_stats.columns = ['domain', 'older_avg_upvotes']

        domain_trend = recent_domain_stats.merge(older_domain_stats, on='domain', how='left')
        domain_trend['domain_trend'] = domain_trend['recent_avg_upvotes'] / domain_trend['older_avg_upvotes'].fillna(
            domain_trend['recent_avg_upvotes'])
        domain_trend = domain_trend[['domain', 'domain_trend']]

        # Merge all domain metrics
        domain_metrics = domain_stats.merge(domain_trend, on='domain', how='left')
        return domain_metrics

    def extract_author_metrics(self, df):
        """Calculate author-based metrics"""
        # Extract author statistics
        author_stats = df.groupby('author')['upvotes'].agg({
            'author_post_count': 'count',
            'author_avg_upvotes': 'mean',
            'author_median_upvotes': 'median',
            'author_max_upvotes': 'max'
        }).reset_index()

        # Calculate author consistency (std deviation of upvotes)
        author_consistency = df.groupby('author')['upvotes'].std().reset_index()
        author_consistency.columns = ['author', 'author_upvote_std']

        # Calculate author trend (last 3 months vs previous)
        now = datetime.now()
        three_months_ago = now - timedelta(days=90)

        recent_posts = df[df['created_at'] > three_months_ago]
        recent_author_stats = recent_posts.groupby('author')['upvotes'].mean().reset_index()
        recent_author_stats.columns = ['author', 'recent_avg_upvotes']

        author_overall_stats = df.groupby('author')['upvotes'].mean().reset_index()
        author_overall_stats.columns = ['author', 'overall_avg_upvotes']

        author_trend = recent_author_stats.merge(author_overall_stats, on='author', how='left')
        author_trend['author_trend'] = author_trend['recent_avg_upvotes'] / author_trend['overall_avg_upvotes']
        author_trend = author_trend[['author', 'author_trend']]

        # Merge all author metrics
        author_metrics = author_stats.merge(author_consistency, on='author', how='left')
        author_metrics = author_metrics.merge(author_trend, on='author', how='left')
        return author_metrics

    def extract_subject_matter(self, df, n_topics=20):
        """Extract subject matter from titles using TF-IDF"""
        # Preprocess titles
        df['clean_title'] = df['title'].str.lower()
        df['clean_title'] = df['clean_title'].apply(lambda x: ' '.join([word for word in re.split(r'\W+', x)
                                                                        if word and word not in self.stopwords]))

        # TF-IDF vectorization
        tfidf = TfidfVectorizer(max_features=n_topics, stop_words='english')
        tfidf_matrix = tfidf.fit_transform(df['clean_title'])
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names_out())

        # Attach topic scores to original DataFrame
        topic_columns = {f'topic_{i}': col for i, col in enumerate(tfidf_df.columns)}
        tfidf_df = tfidf_df.rename(columns=topic_columns)
        tfidf_df.index = df.index

        # Merge with original data
        result_df = pd.concat([df, tfidf_df], axis=1)
        return result_df, tfidf.get_feature_names_out()

    def create_time_features(self, df):
        """Create time-based features"""
        # Basic time components
        df['post_hour'] = df['created_at'].dt.hour
        df['post_day'] = df['created_at'].dt.dayofweek  # 0=Monday, 6=Sunday
        df['post_month'] = df['created_at'].dt.month
        df['post_dayofmonth'] = df['created_at'].dt.day
        df['post_weekend'] = df['post_day'].isin([5, 6]).astype(int)
        df['post_us_business_hours'] = ((df['post_hour'] >= 14) & (df['post_hour'] <= 22) &
                                        (~df['post_weekend'])).astype(int)  # 9am-5pm EST is roughly 14-22 UTC

        # User age at post time (days)
        df['user_age_days'] = (df['created_at'] - df['user_created_at']).dt.days

        # Time since account creation buckets
        df['user_age_bucket'] = pd.cut(
            df['user_age_days'],
            bins=[0, 30, 90, 365, 365 * 2, 365 * 5, 365 * 10, np.inf],
            labels=['<1mo', '1-3mo', '3-12mo', '1-2yr', '2-5yr', '5-10yr', '>10yr']
        )

        # Add quarter
        df['post_quarter'] = df['created_at'].dt.quarter

        return df

    def create_content_features(self, df):
        """Create features from post content"""
        # Title features
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

        # URL features
        df['is_github'] = df['url'].str.contains('github.com', case=False).fillna(False).astype(int)
        df['is_blog'] = df['url'].str.contains('blog|medium.com', case=False).fillna(False).astype(int)
        df['is_video'] = df['url'].str.contains('youtube.com|vimeo|youtu.be', case=False).fillna(False).astype(int)

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
                         'post_weekend', 'post_us_business_hours', 'post_quarter']

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


# Example usage
if __name__ == "__main__":
    # Initialize feature engineer
    db_connection = "postgresql://username:password@localhost:5432/hn_database"
    fe = HNFeatureEngineer(db_connection)

    # Process data
    processed_data = fe.process_data(limit=10000)

    # Save to CSV
    processed_data.to_csv('hn_features.csv', index=False)

    # Display sample features
    print("\nSample data with features:")
    print(processed_data.head())

    # Display feature correlation with upvotes
    numerical_features = processed_data.select_dtypes(include=['number']).columns
    numerical_features = [f for f in numerical_features if f != 'upvotes' and 'topic_' not in f]

    correlations = processed_data[numerical_features + ['upvotes']].corr()['upvotes'].sort_values(ascending=False)
    print("\nFeature correlations with upvotes:")
    print(correlations)