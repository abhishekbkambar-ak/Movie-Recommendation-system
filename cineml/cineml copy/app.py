from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import requests
import os
import pickle
import re

# ===========================
# Load environment variables
# ===========================

load_dotenv()

app = Flask(__name__)
CORS(app)

API_KEY = os.getenv("OMDB_API_KEY")


# ===========================
# Load model files
# ===========================

movies = None
similarity = None

try:

    with open("model/movies.pkl", "rb") as f:
        movies = pickle.load(f)

    with open("model/similarity.pkl", "rb") as f:
        similarity = pickle.load(f)

    print(f"✅ Model loaded successfully")
    print(f"✅ {len(movies)} movies loaded")

except Exception as e:

    print("❌ Model loading error:")
    print(e)


# ===========================
# Fetch movie poster
# ===========================

def fetch_poster(movie_name):

    if not API_KEY:
        return "https://via.placeholder.com/300x450?text=No+Poster"

    try:

        url = (
            f"http://www.omdbapi.com/"
            f"?t={movie_name}&apikey={API_KEY}"
        )

        response = requests.get(
            url,
            timeout=5
        )

        data = response.json()

        if (
            data.get("Response") == "True"
            and data.get("Poster")
            and data["Poster"] != "N/A"
        ):

            return data["Poster"]

    except Exception as e:

        print("Poster error:", e)

    return "https://via.placeholder.com/300x450?text=No+Poster"


# ===========================
# Recommendation logic
# ===========================

def recommend(movie):

    if movies is None or similarity is None:
        return []

    movie = movie.strip().lower()

    escaped_movie = re.escape(movie)

    matched = movies[
        movies["title"]
        .str.lower()
        .str.contains(
            escaped_movie,
            na=False,
            regex=True
        )
    ]

    if matched.empty:
        raise Exception("Movie not found")

    movie_index = matched.index[0]

    distances = similarity[movie_index]

    movies_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    recommendations = []

    for item in movies_list:

        index = item[0]
        score = item[1]

        movie_name = movies.iloc[index]["title"]

        recommendations.append({

            "title": movie_name,

            "poster": fetch_poster(
                movie_name
            ),

            "year": "",

            "rating": "N/A",

            "genres": [],

            "similarity":
            round(
                float(score) * 100,
                1
            )

        })

    return recommendations


# ===========================
# Home page
# ===========================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ===========================
# Recommendation API
# ===========================

@app.route(
    "/api/recommend",
    methods=["POST"]
)
def get_recommendations():

    try:

        data = request.get_json()

        movie = data.get(
            "movie",
            ""
        ).strip()

        if movie == "":

            return jsonify({

                "error":
                "Enter movie name"

            }), 400

        results = recommend(movie)

        return jsonify({

            "query": {

                "title": movie,

                "poster":
                fetch_poster(movie),

                "year": "",

                "rating": "N/A",

                "genres": [],

                "overview": ""

            },

            "recommendations":
            results

        })

    except Exception as e:

        print(
            "Recommendation error:",
            e
        )

        return jsonify({

            "error":
            "Movie not found"

        }), 404


# ===========================
# Status API
# ===========================

@app.route("/api/status")
def status():

    return jsonify({

        "status": "online",

        "model_ready":
        (
            movies is not None
            and similarity is not None
        ),

        "movie_count":
        len(movies)
        if movies is not None
        else 0

    })


# ===========================
# Popular movies API
# ===========================

@app.route("/api/popular")
def popular():

    movie_list = [

        "Inception",
        "Interstellar",
        "The Matrix",
        "Avatar",
        "Titanic"

    ]

    results = []

    for movie in movie_list:

        results.append({

            "title": movie,

            "poster":
            fetch_poster(movie)

        })

    return jsonify(results)


# ===========================
# Suggestions API
# ===========================

@app.route("/api/suggestions")
def suggestions():

    query = request.args.get(
        "q",
        ""
    ).strip().lower()

    if query == "" or movies is None:

        return jsonify([])

    escaped_query = re.escape(query)

    results = movies[
        movies["title"]
        .str.lower()
        .str.contains(
            escaped_query,
            na=False,
            regex=True
        )
    ]["title"].head(8)

    return jsonify(
        results.tolist()
    )


# ===========================
# Run app
# ===========================

if __name__ == "__main__":

    app.run(
        debug=True
    )