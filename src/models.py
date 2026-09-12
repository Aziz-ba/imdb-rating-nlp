"""The three modeling approaches, each behind a builder function.

Only the TF-IDF + Ridge baseline needs no heavy dependencies; the USE and
DistilBERT builders import TensorFlow / PyTorch lazily so the baseline stays
lightweight.
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

def build_tfidf_ridge(max_features: int = 5000, alpha: float = 1.0) -> Pipeline:
    """Sparse bag-of-words baseline: TF-IDF features -> Ridge regression."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=max_features, stop_words="english")),
        ("ridge", Ridge(alpha=alpha)),
    ])

def embed_use(texts):
    """Encode texts with the Universal Sentence Encoder (512-d)."""
    import tensorflow_hub as hub  # lazy import
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    return embed(list(texts)).numpy()

def build_use_dnn(input_dim: int = 512):
    """Dense network on top of USE embeddings."""
    import tensorflow as tf  # lazy import
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(1),
    ])
