# 🎬 CineML — Movie Recommendation System

A production-grade, content-based movie recommendation engine with a Netflix-dark UI.  
Built with Python, Flask, Scikit-learn, TF-IDF, and the TMDB API.

---

## ⚡ Quick Start (Mac — one command)

```bash
chmod +x setup.sh && ./setup.sh
```

That's it. The script handles everything and opens the app at **http://localhost:5000**.

---

## 📁 Project Structure

```
cineml/
├── app.py              ← Flask web server + API routes
├── train.py            ← ML model builder (TF-IDF + cosine similarity)
├── setup.sh            ← One-command Mac setup script
├── requirements.txt    ← Python dependencies
├── .env.example        ← API key template
├── .env                ← Your actual keys (git-ignored)
│
├── model/
│   ├── movies.pkl      ← Trained movie DataFrame
│   ├── similarity.pkl  ← Cosine similarity matrix
│   └── tfidf.pkl       ← Fitted TF-IDF vectorizer
│
├── data/
│   ├── tmdb_5000_movies.csv   ← TMDB dataset (or auto-generated demo)
│   └── tmdb_5000_credits.csv
│
├── templates/
│   └── index.html      ← Netflix-dark HTML UI
│
└── static/
    ├── css/style.css   ← All styling
    └── js/app.js       ← Frontend logic
```

---

## 🧠 How the ML Works

### 1. Feature Engineering
Each movie's metadata is merged into a single "tags" string:
- **Genres** (Action, Thriller…)
- **Keywords** (heist, dream, dystopia…)
- **Top 5 Cast** (slugified names)
- **Director** (slugified name)

### 2. TF-IDF Vectorization
```python
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
vectors = tfidf.fit_transform(df['tags'])
```
Converts text into a 5000-dimension numeric vector per movie.

### 3. Cosine Similarity
```python
similarity = cosine_similarity(vectors)
```
Computes a pairwise similarity matrix (N×N). Values range 0–1.

### 4. Recommendation
```python
def recommend(movie, n=5):
    idx = df[df['title'] == movie].index[0]
    scores = sorted(enumerate(similarity[idx]), key=lambda x: x[1], reverse=True)
    return [df.iloc[i]['title'] for i, _ in scores[1:n+1]]
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/` | Main UI |
| `POST` | `/api/recommend` | Get top-5 recommendations |
| `GET`  | `/api/popular` | TMDB trending movies |
| `GET`  | `/api/search_suggest?q=` | Autocomplete movie titles |
| `GET`  | `/api/status` | Model & API health check |

### Example — POST /api/recommend
```bash
curl -X POST http://localhost:5000/api/recommend \
  -H "Content-Type: application/json" \
  -d '{"movie": "The Dark Knight"}'
```
Response:
```json
{
  "query": { "title": "The Dark Knight", "rating": 9.0, "year": "2008", ... },
  "recommendations": [
    { "title": "Batman Begins", "similarity": 96.4, "rating": 8.2, ... },
    ...
  ]
}
```

---

## 🎬 TMDB API Key (optional)

Without it: recommendations still work, no movie posters or trending section.  
With it: full posters, ratings, overviews, genres from TMDB.

1. Go to https://www.themoviedb.org/settings/api
2. Register for a free account
3. Copy your API key (v3 auth)
4. Add to `.env`: `TMDB_API_KEY=your_key_here`

---

## 📊 Full TMDB 5000 Dataset (optional)

The app ships with a 110-movie demo dataset. For the full 5000-movie corpus:

**Option A — Kaggle CLI**
```bash
pip install kaggle
# Place kaggle.json in ~/.kaggle/
kaggle datasets download -d tmdb/tmdb-movie-metadata --unzip -p data/
python train.py   # retrain
```

**Option B — Manual**
1. Download from: https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata
2. Place both CSVs in `./data/`
3. Run: `python train.py`

---

## 🛠 Manual Setup (without setup.sh)

```bash
# 1. Create venv
python3 -m venv venv && source venv/bin/activate

# 2. Install deps
pip install -r requirements.txt

# 3. Set up keys
cp .env.example .env
# Edit .env and add your TMDB_API_KEY

# 4. Train model
python train.py

# 5. Run app
python app.py
# Open http://localhost:5000
```

---

## 🔧 Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.8+ |
| Web Framework | Flask 3.0 |
| ML — Vectorization | Scikit-learn TfidfVectorizer |
| ML — Similarity | Scikit-learn cosine_similarity |
| NLP Preprocessing | NLTK PorterStemmer |
| Data Manipulation | Pandas, NumPy |
| External API | TMDB API v3 |
| Frontend | HTML5, CSS3, Vanilla JS |
| Fonts | Bebas Neue, Barlow (Google Fonts) |

---

## 📚 References

- [TMDB 5000 Movie Dataset — Kaggle](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)
- [TMDB API Docs](https://developers.themoviedb.org/3)
- [Scikit-learn TF-IDF](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
- [Towards Data Science — Recommendation Systems](https://towardsdatascience.com)
- [Krish Naik — Movie Recommendation System (YouTube)](https://www.youtube.com/results?search_query=movie+recommendation+system+krish+naik)

---

## 🪪 License
MIT — free to use, modify, and deploy.
