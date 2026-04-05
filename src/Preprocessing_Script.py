import string
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

def clean_text(text):
    """
    Lowercases, removes punctuation/digits, and filters out stop words.
    """
    text = str(text).lower()
    text = ''.join([c for c in text if c not in string.punctuation])
    text = ''.join([c for c in text if not c.isdigit()])
    words = text.split()
    words = [w for w in words if w not in ENGLISH_STOP_WORDS and len(w) > 2]
    return ' '.join(words)
