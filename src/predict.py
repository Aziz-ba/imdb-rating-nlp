"""Predict an IMDb rating for a plot description using a saved TF-IDF model.

    python src/train.py --data data/IMDB-Movie-Data.csv --save model.joblib
    python src/predict.py model.joblib "A thief who steals corporate secrets through dream-sharing tech."
"""
import sys
import joblib
from preprocess import clean_text

def main():
    if len(sys.argv) < 3:
        print("usage: python predict.py <model.joblib> \"<description>\""); sys.exit(1)
    model = joblib.load(sys.argv[1])
    text = clean_text(sys.argv[2])
    rating = float(model.predict([text])[0])
    print(f"Predicted IMDb rating: {rating:.2f} / 10")

if __name__ == "__main__":
    main()
