from Preprocessing_Script import clean_text
from Model_Script import get_pipeline
import pandas as pd
import joblib
from Preprocessing_Script import clean_text
from Model_Script import get_pipeline

def run_project():
    # 1. Load Data
    train = pd.read_csv('../data/development.csv')
    eval_data = pd.read_csv('../data/evaluation.csv')

    # 2. Preprocess
    print("Preprocessing text...")
    for df in [train, eval_data]:
        df['clean_title'] = df['title'].apply(clean_text)
        df['clean_article'] = df['article'].apply(clean_text)

    # 3. Train
    X_train_full = train[['clean_title','clean_article', 'source', 'page_rank']]
    y_train_full = train['label']

    pipeline = get_pipeline()
    print("Fitting model...")
    pipeline.fit(X_train_full, y_train_full)

    # 4. Save Model
    joblib.dump(pipeline, '../models/news_classifier.pkl')

    # 5. Generate Submission
    print("Generating submission.csv...")
    eval_features = eval_data[['clean_title', 'clean_article', 'source', 'page_rank']]
    eval_preds = pipeline.predict(eval_features)
    
    submission = pd.DataFrame({'Id': eval_data['Id'], 'Predicted': eval_preds})
    submission.to_csv('../submission.csv', index=False)
    print("Done!")

if __name__ == "__main__":
    run_project()
