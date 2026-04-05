# News Classification ML 📰

This project classifies 100k news articles into 10 categories with an **82.5% Macro F1-score using a Logistic Regression classifier.**.

##  Structure
- `src/`: Contains the preprocessing logic, model architecture, and main execution script.
- `data/`: Placeholder for `development.csv` and `evaluation.csv`.
- `models/`: Where the trained `.pkl` file is saved.

##  How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. The dataset is located at:
https://drive.google.com/file/d/1dWNUwC47MfPs7mxsP0_e2rmUY4_AUDCk/view?usp=sharing
Within the archive, you will find the following elements:
• development.csv (development set): a comma-separated values file containing the records from
the development set. This portion does have the label column, which you should use to train and
validate your models.
• evaluation.csv (evaluation set): a comma-separated values file containing the records corresponding
to the evaluation set. This portion does not have the label column
3. Place your CSV data in the `/data` folder.
4. Run the pipeline: `python src/main.py`
