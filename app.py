import streamlit as st
import pandas as pd
import ast
import requests
import urllib.parse
from io import BytesIO

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------
# PAGE CONFIG
# -----------------------
st.set_page_config(
    page_title="Movie Recommendation System",
    layout="wide"
)

# -----------------------
# LOAD DATA (CACHED)
# -----------------------
@st.cache_data
def load_data():
    movies = pd.read_csv(
        "tmdb_5000_movies.csv",
        encoding="latin-1",
        engine="python",
        on_bad_lines="skip"
    )

    
   movies = movies[['movie_id','title','overview','genres','keywords']]
   movies.dropna(inplace=True)

    return movies

movies = load_data()

# -----------------------
# PREPROCESSING
# -----------------------
def convert(text):
    return [i['name'] for i in ast.literal_eval(text)]

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)

movies['tags'] = movies['genres'] + movies['keywords']
movies['tags'] = movies['tags'].apply(lambda x: " ".join(x))

new = movies[['movie_id', 'title', 'tags']]

# -----------------------
# VECTOR + SIMILARITY (CACHED)
# -----------------------
@st.cache_data
def compute_similarity():
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(new['tags']).toarray()
    similarity = cosine_similarity(vectors)
    return cv, vectors, similarity

cv, vector, similarity = compute_similarity()

# -----------------------
# RECOMMENDER
# -----------------------
def recommend_by_keywords(keyword_string):
    keyword_vector = cv.transform([keyword_string])
    scores = cosine_similarity(keyword_vector, vector)
    top_indices = scores.flatten().argsort()[-5:][::-1]
    return [new.iloc[i].title for i in top_indices]

# -----------------------
# TMDB API (SAFE + CACHED)
# -----------------------
API_KEY = "b7bb86b044a3125c449ccd03a5d51b32"

@st.cache_data(show_spinner=False)
def get_movie_id(movie_name):
    try:
        query = urllib.parse.quote(movie_name)
        url = f"https://api.themoviedb.org/3/search/movie?api_key={API_KEY}&query={query}"
        res = requests.get(url, timeout=5)
        res.raise_for_status()
        data = res.json()

        for m in data.get("results", []):
            if m.get("poster_path"):
                return m["id"]
        return None
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def fetch_poster(movie_id):
    if movie_id is None:
        return None
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}"
        res = requests.get(url, timeout=5)
        res.raise_for_status()
        data = res.json()

        poster_path = data.get("poster_path")
        if not poster_path:
            return None

        image_url = "https://image.tmdb.org/t/p/w500" + poster_path
        img_res = requests.get(image_url, timeout=5)
        img_res.raise_for_status()

        return BytesIO(img_res.content)
    except Exception:
        return None

# -----------------------
# STREAMLIT UI
# -----------------------
st.title("🎬 Movie Recommendation System")
st.write("Keyword-based movie recommendation with TMDB posters")

keyword = st.text_input(
    "Enter a keyword (love, romance, ghost, thriller, etc)"
)

if st.button("Recommend"):
    results = recommend_by_keywords(keyword)

    st.subheader("Recommended Movies")
    cols = st.columns(5)

    for idx, movie in enumerate(results):
        movie_id = get_movie_id(movie)
        poster = fetch_poster(movie_id)

        with cols[idx]:
            st.caption(movie)
            if poster is not None:
                st.image(poster, width=200)
            else:
                st.markdown(
                    f"""
                    <div style="width:200px;height:300px;
                    display:flex;align-items:center;justify-content:center;
                    background-color:#f0f0f0;border-radius:10px;
                    text-align:center;font-weight:bold;">
                    {movie}
                    </div>
                    """,
                    unsafe_allow_html=True
                )


