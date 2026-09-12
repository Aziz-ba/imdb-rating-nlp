# Data

- `sample_movies.csv` - a tiny bundled sample (real films & approximate IMDb ratings) so the
  pipeline runs out of the box. It is **only for smoke-testing** - far too small for meaningful metrics.

## Full dataset

The reported results use the Kaggle **IMDB-Movie-Data** dataset (~1000 films):

```python
import kagglehub
path = kagglehub.dataset_download("PromptCloudHQ/imdb-data")
# -> IMDB-Movie-Data.csv  (columns include Title, Description, Rating)
```

Then:

```bash
python src/train.py --data path/to/IMDB-Movie-Data.csv --model tfidf --save model.joblib
```
