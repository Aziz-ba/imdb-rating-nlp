"""Train and evaluate a model that predicts IMDb rating from plot text.

Usage:
    python src/train.py --data data/sample_movies.csv --model tfidf
    python src/train.py --data data/IMDB-Movie-Data.csv --model tfidf --save model.joblib
"""
import argparse
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from preprocess import load_data
from models import build_tfidf_ridge

def evaluate(y_true, y_pred):
    return {
        "MAE": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "R2": round(float(r2_score(y_true, y_pred)), 4),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", choices=["tfidf"], default="tfidf",
                    help="tfidf baseline (USE/DistilBERT live in models.py and need TF/PyTorch)")
    ap.add_argument("--save")
    ap.add_argument("--test-size", type=float, default=0.2)
    args = ap.parse_args()

    df = load_data(args.data)
    print(f"Loaded {len(df)} rows.")
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean"], df["rating"], test_size=args.test_size, random_state=42
    )

    model = build_tfidf_ridge()
    model.fit(X_train, y_train)
    metrics = evaluate(y_test, model.predict(X_test))
    print("Metrics:", json.dumps(metrics))

    if args.save:
        import joblib
        joblib.dump(model, args.save)
        print("Saved model to", args.save)

if __name__ == "__main__":
    main()
