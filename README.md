# 🎥 Predicting IMDb Ratings from Text — an NLP benchmark

Can you predict a movie's **IMDb rating** from nothing but its **plot description**? This project answers that question by building and comparing **three progressively more powerful NLP models** on the same task.

A single regression target (`IMDB_Rating`), one text feature (the movie description), and three modeling philosophies put head to head.

---

## 🧪 The three approaches

| # | Approach | Idea | Libraries |
|---|----------|------|-----------|
| 1 | **TF-IDF + Ridge Regression** | Classic sparse bag-of-words baseline | scikit-learn |
| 2 | **Universal Sentence Encoder + DNN** | Dense 512-d semantic embeddings feeding a Keras neural net | TensorFlow Hub, Keras |
| 3 | **DistilBERT (fine-tuned)** | Transformer contextual embeddings for regression | 🤗 Transformers, PyTorch |

The notebook covers the full workflow: **text cleaning → EDA → vectorization/embedding → training → evaluation**, so you can see exactly how much each jump in model sophistication buys you.

---

## 🔬 Pipeline

1. **Load** the [IMDB Movie dataset](https://www.kaggle.com/datasets/PromptCloudHQ/imdb-data) via `kagglehub`.
2. **Clean** the text (lowercase, strip punctuation) and keep `Title`, `Description`, `Rating`.
3. **Explore** the rating distribution and text length.
4. **Model** with the three approaches above and compare error.

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Hugging Face](https://img.shields.io/badge/🤗_Transformers-FFD21E?style=flat-square)

pandas · numpy · matplotlib · seaborn · scikit-learn · TensorFlow/Keras · TF-Hub (Universal Sentence Encoder) · PyTorch · DistilBERT

---

## 🚀 Run it

```bash
pip install -r requirements.txt
jupyter notebook notebooks/imdb_rating_nlp.ipynb
```

The dataset is pulled automatically with `kagglehub` (you'll need Kaggle credentials configured locally).

---

## 📚 What this project demonstrates

- End-to-end **NLP regression** workflow
- Text preprocessing and exploratory analysis
- **Benchmarking** classic ML vs. deep learning vs. transformers on one task
- Practical use of **TF-IDF, sentence embeddings, and DistilBERT**

---

## 📄 License

Released under the [MIT License](LICENSE).
