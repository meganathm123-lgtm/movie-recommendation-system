import streamlit as st
import pandas as pd
import requests
import urllib.parse

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_css():
    with open("styles/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Movie Recommendation System", page_icon="🎥", layout="wide")

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    return pd.read_csv("final_movies.csv")

movies = load_data()

# ---------------- MODEL ----------------
@st.cache_data
def build_model(df):
    cv = CountVectorizer(max_features=5000, stop_words="english")
    vectors = cv.fit_transform(df["tags"]).toarray()
    similarity = cosine_similarity(vectors)
    return cv, vectors, similarity

cv, vectors, similarity = build_model(movies)

# ---------------- RECOMMENDER ----------------
def recommend_by_keywords(keyword, n=5):
    keyword_vector = cv.transform([keyword])
    scores = cosine_similarity(keyword_vector, vectors).flatten()
    indices = scores.argsort()[-n:][::-1]
    return movies.iloc[indices]["title"].tolist()

# ---------------- TMDB ----------------
API_KEY = st.secrets["TMDB_API_KEY"]

@st.cache_data(show_spinner=False)
def fetch_poster(movie_name):
    try:
        query = urllib.parse.quote(movie_name)
        url = f"https://api.themoviedb.org/3/search/movie?api_key={API_KEY}&query={query}"
        data = requests.get(url, timeout=5).json()

        if not data.get("results"):
            return None

        poster_path = data["results"][0].get("poster_path")
        if not poster_path:
            return None

        return "https://image.tmdb.org/t/p/w500" + poster_path
    except:
        return None

# ---------------- UI ----------------
st.markdown("""
<h1>🎬 Movie Recommendation System</h1>
<p style="font-size:1.1rem; opacity:0.8; max-width:700px;">
Discover movies by mood, genre, or vibe.  
Type a keyword and let the system do the magic ✨
</p>
""", unsafe_allow_html=True)


keyword = st.text_input(
    "Enter a keyword (love, romance, ghost, thriller, etc)",
    key="keyword_input"
)

clicked = st.button("Recommend")

if clicked:
    if keyword.strip() == "":
        st.warning("Please enter a keyword")
    else:
        results = recommend_by_keywords(keyword)

        st.subheader("🍿 Recommended Movies")
        cols = st.columns(5)

        for i, movie in enumerate(results):
            poster = fetch_poster(movie)

            with cols[i]:
                if poster:
                    st.image(poster, use_container_width=True)
                else:
                    st.image(
                        "https://via.placeholder.com/300x450?text=No+Poster",
                        use_container_width=True
                    )

                st.markdown(f'<div class="movie-title">{movie}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
