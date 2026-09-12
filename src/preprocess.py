"""Data loading and text cleaning for the IMDb rating task."""
import re
import string
import pandas as pd

def clean_text(text: str) -> str:
    """Lowercase and strip punctuation."""
    text = str(text).lower()
    return re.sub(f"[{re.escape(string.punctuation)}]", "", text)

def load_data(csv_path: str) -> pd.DataFrame:
    """Load a movies CSV and return columns: description, rating.

    Accepts the Kaggle 'IMDB-Movie-Data.csv' schema (Title/Description/Rating)
    or the bundled sample (title/description/rating).
    """
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.rename(columns={"imdb_rating": "rating"})
    df = df[["description", "rating"]].dropna()
    df = df[df["description"].str.len() > 10]
    df["clean"] = df["description"].apply(clean_text)
    return df.reset_index(drop=True)
