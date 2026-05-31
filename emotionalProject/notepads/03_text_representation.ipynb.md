```py
# cell 1: toy example - bag of words (bow) vs tf-idf
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# create a tiny toy corpus of texts
corpus = [
    "the movie is good",
    "the movie is bad",
    "the movie is very very good"
]

print("original corpus:")
for i, doc in enumerate(corpus):
    print(f"doc {i}: '{doc}'")

# 1. bag of words 
# # just count times of entering each word
vectorizer_bow = CountVectorizer()
bow_matrix = vectorizer_bow.fit_transform(corpus)

# get the vocabulary (which word became which column)
vocab = vectorizer_bow.get_feature_names_out()

df_bow = pd.DataFrame(bow_matrix.toarray(), columns=vocab)
print("\n--- bag of words (count matrix) ---")
print(df_bow)

# 2. tf-idf (term frequency - inverse document frequency)
# penalizes frequent words, gives weight to rare and important words
vectorizer_tfidf = TfidfVectorizer()
tfidf_matrix = vectorizer_tfidf.fit_transform(corpus)

df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=vocab)
print("\n--- tf-idf (weight matrix) ---")
print(df_tfidf.round(3))
```


```
original corpus:
doc 0: 'the movie is good'
doc 1: 'the movie is bad'
doc 2: 'the movie is very very good'

--- bag of words (count matrix) ---
   bad  good  is  movie  the  very
0    0     1   1      1    1     0
1    1     0   1      1    1     0
2    0     1   1      1    1     2

--- tf-idf (weight matrix) ---
     bad   good     is  movie    the   very
0  0.000  0.597  0.463  0.463  0.463  0.000
1  0.699  0.000  0.413  0.413  0.413  0.000
2  0.000  0.321  0.249  0.249  0.249  0.843
```

```py
# cell 2: tf-idf on a real dataset sample (model a)
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

print("loading a 1000-row sample from model a...")
# we only load a small chunk to avoid crashing the ram
df_sample = pd.read_csv('data/processed/model_a_train_cleaned.csv', nrows=1000)

# ensure no null values crept in
df_sample = df_sample.dropna(subset=['clean_text'])

# initialize standard tf-idf without any limits
vectorizer = TfidfVectorizer()

# fit and transform the text into a matrix
tfidf_matrix = vectorizer.fit_transform(df_sample['clean_text'])

print(f"\nsample size: {df_sample.shape[0]} documents")
print(f"matrix shape: {tfidf_matrix.shape}")
print(f"vocabulary size (unique words): {len(vectorizer.get_feature_names_out())}")
```

```
loading a 1000-row sample from model a...

sample size: 1000 documents
matrix shape: (1000, 11830)
vocabulary size (unique words): 11830
```

```py
# cell 3: applying tf-idf with vocabulary limits
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

print("testing limited tf-idf on the same sample...")

# min_df=5: word must appear in at least 5 documents
# max_features=5000: keep only the top 5000 most frequent words
# ngram_range=(1, 2) means: extract single words AND pairs of two adjacent words
vectorizer_limited = TfidfVectorizer(min_df=5, max_features=5000, ngram_range=(1, 2))

tfidf_matrix_limited = vectorizer_limited.fit_transform(df_sample['clean_text'])

print(f"\nnew matrix shape: {tfidf_matrix_limited.shape}")
print(f"new vocabulary size: {len(vectorizer_limited.get_feature_names_out())}")

# let's look at some randomly selected features that survived the cut
import random
vocab = vectorizer_limited.get_feature_names_out()
print("\nrandom sample of kept words:")
print(random.sample(list(vocab), min(20, len(vocab))))
```

```
testing limited tf-idf on the same sample...

new matrix shape: (1000, 2706)
new vocabulary size: 2706

random sample of kept words:
['robert', 'person', 'terror', 'mild', 'film know', 'crash', 'performance not', 'film don', 'costume', 'watchable', 'sue', 'minor', 'young boy', 'exactly', 'film good', 'indication', 'dialogue', 'island', 'fourth', 'want watch']
```

note on dataset splitting strategy (model a)
I decided to use the already cleaned `model_a_train_cleaned.csv` (which contains imdb train + goemotions neutral) and split it internally using `train_test_split`. the original raw `imdb_test.csv` remains untouched in the `raw` folder. This approach provides more than enough data (~30,000 rows) for our baseline algorithms without requiring an additional preprocessing pass.

```py
# cell 4: proper train-test split and tf-idf vectorization (model a)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse as sp
import joblib
import os

print("loading full cleaned dataset for model a...")
df_a = pd.read_csv('data/processed/model_a_train_cleaned.csv')

# drop any rows that might have become empty (nan) after pandas loaded the csv
df_a = df_a.dropna(subset=['clean_text'])

# 1. train-test split (80% train, 20% test)
# stratify=df_a['label'] ensures both sets have the same ratio of positive/negative reviews
x_train, x_test, y_train, y_test = train_test_split(
    df_a['clean_text'], 
    df_a['label'], 
    test_size=0.2, 
    random_state=42, 
    stratify=df_a['label']
)

print(f"train size: {x_train.shape[0]}, test size: {x_test.shape[0]}")

# 2. tf-idf vectorization
print("fitting tf-idf vectorizer on training data...")
# max_features=10000 is a standard robust baseline for imdb
# ngram_range=(1, 2) means: extract single words AND pairs of two adjacent words
vectorizer_a = TfidfVectorizer(min_df=5, max_features=10000, ngram_range=(1, 2))

# we FIT the vocabulary only on x_train, but TRANSFORM both
x_train_tfidf = vectorizer_a.fit_transform(x_train)
x_test_tfidf = vectorizer_a.transform(x_test)

print(f"final matrix shape (train): {x_train_tfidf.shape}")

# 3. save artifacts for phase 4 and production
os.makedirs('models/vectorizers', exist_ok=True)
os.makedirs('data/features', exist_ok=True)

print("saving vectorizer and sparse matrices...")
# save the vectorizer object so the telegram bot can use the exact same vocabulary later
joblib.dump(vectorizer_a, 'models/vectorizers/tfidf_model_a.pkl')

# save the resulting matrices in a compressed sparse format (.npz)
sp.save_npz('data/features/model_a_x_train.npz', x_train_tfidf)
sp.save_npz('data/features/model_a_x_test.npz', x_test_tfidf)

# save the labels
y_train.to_csv('data/features/model_a_y_train.csv', index=False)
y_test.to_csv('data/features/model_a_y_test.csv', index=False)

print("model a vectorization complete!")
```

```
loading full cleaned dataset for model a...
train size: 31247, test size: 7812
fitting tf-idf vectorizer on training data...
final matrix shape (train): (31247, 10000)
saving vectorizer and sparse matrices...
model a vectorization complete!
```

```py
# cell 5: train-test split and tf-idf for models b and c
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse as sp
import joblib

# --- model b (emotions) ---
print("processing model b (emotions)...")
df_b = pd.read_csv('data/processed/model_b_train_cleaned.csv').dropna(subset=['clean_text'])

# correct 9 emotion columns for target y
emotion_cols = ['sadness', 'anger', 'fear', 'disgust', 'anticipation', 'joy', 'surprise', 'gratitude', 'love']
x_b = df_b['clean_text']
y_b = df_b[emotion_cols]

# standard split (stratification is complex for multi-label, so we omit it here)
x_train_b, x_test_b, y_train_b, y_test_b = train_test_split(
    x_b, y_b, test_size=0.2, random_state=42
)

# vectorization
vectorizer_b = TfidfVectorizer(min_df=5, max_features=10000, ngram_range=(1, 2))
x_train_tfidf_b = vectorizer_b.fit_transform(x_train_b)
x_test_tfidf_b = vectorizer_b.transform(x_test_b)

# save artifacts
joblib.dump(vectorizer_b, 'models/vectorizers/tfidf_model_b.pkl')
sp.save_npz('data/features/model_b_x_train.npz', x_train_tfidf_b)
sp.save_npz('data/features/model_b_x_test.npz', x_test_tfidf_b)
y_train_b.to_csv('data/features/model_b_y_train.csv', index=False)
y_test_b.to_csv('data/features/model_b_y_test.csv', index=False)
print("model b artifacts saved successfully.\n")

# --- model c (formality) ---
print("processing model c (formality)...")
df_c = pd.read_csv('data/processed/model_c_train_cleaned.csv').dropna(subset=['clean_text'])

x_c = df_c['clean_text']
y_c = df_c['label']

# stratified split for binary/multiclass target
x_train_c, x_test_c, y_train_c, y_test_c = train_test_split(
    x_c, y_c, test_size=0.2, random_state=42, stratify=y_c
)

# vectorization
vectorizer_c = TfidfVectorizer(min_df=5, max_features=10000, ngram_range=(1, 2))
x_train_tfidf_c = vectorizer_c.fit_transform(x_train_c)
x_test_tfidf_c = vectorizer_c.transform(x_test_c)

# save artifacts
joblib.dump(vectorizer_c, 'models/vectorizers/tfidf_model_c.pkl')
sp.save_npz('data/features/model_c_x_train.npz', x_train_tfidf_c)
sp.save_npz('data/features/model_c_x_test.npz', x_test_tfidf_c)
y_train_c.to_csv('data/features/model_c_y_train.csv', index=False)
y_test_c.to_csv('data/features/model_c_y_test.csv', index=False)
print("model c artifacts saved successfully.")
```


dataset splitting and stratification notes
**1. splitting strategy (model a):**
I use `train_test_split` on the `model_a_train_cleaned.csv` file. The original `imdb_test.csv` in the `raw` folder remains untouched. This provides ~30,000 rows, which is sufficient for training and evaluating baseline models.
**2. stratification (`stratify=y`):**
- **models a & c:** used to maintain the original class proportions in both train and test sets. This prevents one set from having an unfair concentration of a specific class (e.g., only positive reviews).
- **model b:** not used. `scikit-learn` does not natively support stratification for multi-label data (9 emotion columns). Given the large dataset size, random splitting is acceptable for a baseline.
**3. tf-idf implementation:**
- **fit_transform(x_train):** the vectorizer learns the vocabulary and idf weights strictly from training data.
- **transform(x_test):** the vectorizer converts test data into numbers using the previously learned vocabulary. Any word present in test but not in train is ignored to prevent data leakage.

```py
# cell 6: visualizing tf-idf results
import pandas as pd

def get_top_tfidf_words(vectorizer, matrix, row_index, top_n=10):
    """extracts top words and their weights for a specific row."""
    feature_names = vectorizer.get_feature_names_out()
    row_data = matrix.getrow(row_index).toarray().flatten()
    top_indices = row_data.argsort()[-top_n:][::-1]
    # converting to standard python float for cleaner print output
    return [(feature_names[i], round(float(row_data[i]), 4)) for i in top_indices if row_data[i] > 0]

# find a longer text for model a (at least 15 words)
best_row_a = 0
for i in range(100):
    if len(str(x_train.iloc[i]).split()) > 10:
        best_row_a = i
        break

print("--- model a: sample row visualization ---")
print(f"text: {x_train.iloc[best_row_a][:250]}...")
print(f"label: {y_train.iloc[best_row_a]}")
print(f"top tf-idf weights: {get_top_tfidf_words(vectorizer_a, x_train_tfidf, best_row_a)}")

# find a longer text for model b
best_row_b = 0
for i in range(100):
    if len(str(x_train_b.iloc[i]).split()) > 6:
        best_row_b = i
        break

print("\n--- model b: sample row visualization ---")
print(f"text: {x_train_b.iloc[best_row_b][:250]}...")
print(f"active emotions: {y_train_b.iloc[best_row_b][y_train_b.iloc[best_row_b] == 1].index.tolist()}")
print(f"top tf-idf weights: {get_top_tfidf_words(vectorizer_b, x_train_tfidf_b, best_row_b)}")
```

```
--- model a: sample row visualization ---
text: notch columbo begin end particularly like interaction columbo killer ruth gordon avid columbo fan t recall doesn t set killer end episode s try determine correct sequence box message nephew leave finally dawn music episode good one...
label: 1
top tf-idf weights: [('columbo', 0.5889), ('killer', 0.2387), ('episode', 0.2249), ('avid', 0.2022), ('good one', 0.1957), ('ruth', 0.187), ('nephew', 0.1862), ('determine', 0.1758), ('begin end', 0.1728), ('dawn', 0.1714)]

--- model b: sample row visualization ---
text: soon mental hospital lol thank recommend love...
active emotions: ['joy', 'gratitude', 'love']
top tf-idf weights: [('recommend', 0.4671), ('hospital', 0.4406), ('lol thank', 0.4406), ('mental', 0.407), ('soon', 0.3469), ('lol', 0.2104), ('love', 0.1829), ('thank', 0.1726)]
```


**adding n-grams to the vectorization pipeline**
By default, tf-idf treats every word as an independent feature (unigrams). This approach completely loses the context of negation, like "not good" or "never happy". To fix this, I updated our vectorizers with the `ngram_range=(1, 2)` parameter.

**Key points of this update:**
* **Mechanism:** The algorithm uses a sliding window to capture both single words and pairs of adjacent words (bigrams).
* **Vocabulary Quality:** Since I already removed "noisy" stop words in stage 2, our bigrams are dense with meaning (e.g., "hero terrible" instead of "the hero was terrible").
* **Complexity Control:** We kept `max_features=10000` to ensure the model stays lightweight. The vectorizer now selects the top 10,000 most statistically significant features from the combined pool of words and pairs.
* **Impact:** This is especially critical for **Model A (Sentiment)** and **Model C (Formality)**, where phrases like "not bad" or "kind regards" are much stronger signals than individual words.



**phase 3.1 conclusion: text representation**
**what I accomplished:**
* **vectorization strategy:** transitioned from raw text to numerical sparse matrices using `tfidfvectorizer`.
* **context preservation:** implemented bigrams (`ngram_range=(1, 2)`) to capture critical multi-word meanings (e.g., "not good", "kind regards"), leveraging our curated stop-word exceptions.
* **dimensionality control:** capped vocabulary at `max_features=10000` and applied `min_df=5` to filter noise and prevent memory overflow.
* **data leakage prevention:** strictly split data into train and test sets *before* fitting the vectorizer. the test set was only transformed, ensuring our models will be evaluated on truly unseen data.
**generated artifacts:**
* `models/vectorizers/`: 3 fitted tf-idf objects (`.pkl`) ready for the future telegram bot.
* `data/features/`: 6 sparse matrices (`.npz`) containing `x_train` and `x_test` for models a, b, and c.
* `data/features/`: 6 csv files containing `y_train` and `y_test` target labels.