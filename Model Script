from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

def get_pipeline():
    title_tfidf = TfidfVectorizer(
        max_features=16000, ngram_range=(1,3), min_df=3, max_df=0.85, 
        sublinear_tf=True, smooth_idf=True, norm='l2'
    )
    article_tfidf = TfidfVectorizer(
        max_features=28000, ngram_range=(1,3), min_df=5, max_df=0.8, 
        sublinear_tf=True, smooth_idf=True, norm='l2'
    )
    
    metadata_pipeline = ColumnTransformer(transformers=[
        ('source', OneHotEncoder(handle_unknown='ignore'), ['source']),
        ('pagerank', StandardScaler(), ['page_rank']),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('title', title_tfidf, 'clean_title'),
        ('article', article_tfidf, 'clean_article'),
        ('meta', metadata_pipeline, ['source', 'page_rank'])
    ])

    return Pipeline([
        ('features', preprocessor),
        ('clf', LogisticRegression(
            C=0.8, penalty='l2', solver='liblinear', max_iter=2000, 
            dual=True, class_weight='balanced', random_state=42
        ))
    ])
