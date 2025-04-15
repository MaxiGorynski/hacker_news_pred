import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import psycopg2
from sqlalchemy import create_engine
from datetime import datetime, timedelta

# Database connection
# Replace with your actual connection details
db_connection_string = "postgresql://username:password@localhost:5432/hn_database"
engine = create_engine(db_connection_string)

# Query to extract raw data
query = """
SELECT 
    posts.id,
    posts.title,
    posts.url,
    COALESCE(posts.domain, 'unknown') as domain,
    posts.author,
    posts.created_at,
    posts.score as upvotes,
    users.karma as author_karma,
    users.created_at as user_created_at
FROM 
    posts
LEFT JOIN 
    users ON posts.author = users.username
WHERE 
    posts.created_at > NOW() - INTERVAL '1 year'
    AND posts.score IS NOT NULL
ORDER BY 
    posts.created_at DESC
LIMIT 10000;
"""

# Load data
df = pd.read_sql(query, engine)

# Basic data inspection
print(f"Dataset shape: {df.shape}")
print("\nData types:")
print(df.dtypes)
print("\nMissing values:")
print(df.isnull().sum())
print("\nSummary statistics:")
print(df.describe())

# Feature engineering - Time components
df['post_hour'] = df['created_at'].dt.hour
df['post_day'] = df['created_at'].dt.day_name()
df['post_month'] = df['created_at'].dt.month_name()
df['post_weekend'] = df['created_at'].dt.weekday >= 5

# User features
df['user_age_days'] = (df['created_at'] - df['user_created_at']).dt.days

# Content features
df['title_length'] = df['title'].str.len()
df['has_question'] = df['title'].str.contains('\?').astype(int)

# Visualizations
plt.figure(figsize=(15, 10))

# Distribution of upvotes
plt.subplot(2, 2, 1)
sns.histplot(df['upvotes'], kde=True)
plt.title('Distribution of Upvotes')
plt.xscale('log')

# Upvotes by day of week
plt.subplot(2, 2, 2)
sns.boxplot(x='post_day', y='upvotes', data=df, order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
plt.title('Upvotes by Day of Week')
plt.xticks(rotation=45)

# Upvotes by hour of day
plt.subplot(2, 2, 3)
sns.boxplot(x='post_hour', y='upvotes', data=df)
plt.title('Upvotes by Hour of Day')

# Title length vs upvotes
plt.subplot(2, 2, 4)
sns.scatterplot(x='title_length', y='upvotes', data=df, alpha=0.5)
plt.title('Title Length vs Upvotes')

plt.tight_layout()
plt.savefig('hn_upvotes_eda.png')

# Top domains analysis
top_domains = df.groupby('domain')['upvotes'].agg(['count', 'mean', 'median', 'std']).sort_values(by='count', ascending=False)
print("\nTop 20 domains by post count:")
print(top_domains.head(20))

# Top authors analysis
top_authors = df.groupby('author')['upvotes'].agg(['count', 'mean', 'median', 'std']).sort_values(by='count', ascending=False)
print("\nTop 20 authors by post count:")
print(top_authors.head(20))

# Correlation analysis
correlation_features = df[['upvotes', 'author_karma', 'title_length', 'user_age_days', 'post_hour']]
correlation = correlation_features.corr()
print("\nFeature correlations:")
print(correlation['upvotes'].sort_values(ascending=False))

# Save processed data
df.to_csv('hn_processed_data.csv', index=False)