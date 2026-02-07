import streamlit as st
import pandas as pd
import requests
import urllib.parse
from io import BytesIO

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------
# PAGE CONFIG
# -----------------------
st.set_page_config(page_title="Movie Recommendation System", layout="wide")

# -----------------------
# LOAD DATA
# -----------------------
@st.cache_data
def load_data():
    return pd.read_csv("final_movies(1).csv")

movies = load_data()

# -----------------------
# VECTORIZE
# -----------------------
cv = CountVectorizer(max_features=5000, stop_words="english")
vectors = cv.fit_transform(movies["tags"]).toarray()
similarity = cosine_similarity(vectors)

# -----------------------
# RECOMMENDER
# -----------------------
def recommend_by_keywords(keyword):
    keyword_vec = cv.transform([keyword])
    scores = cosine_similarity(keyword_vec, vectors)
    top_indices = scores.flatten().argsort()[-5:][::-1]
    return movies.iloc[top_indices][["movie_id", "title"]]

# -----------------------
# TMDB API
# -----------------------
API_KEY = st.secrets["TMDB_API_KEY"]

@st.cache_data(show_spinner=False)
def fetch_poster(title):
    try:
        query = urllib.parse.quote(title)
        url = f"https://api.themoviedb.org/3/search/movie?api_key={API_KEY}&query={query}"
        data = requests.get(url, timeout=5).json()

        for m in data.get("results", []):
            if m.get("poster_path"):
                return "https://image.tmdb.org/t/p/w500" + m["poster_path"]
        return None
    except:
        return None

# -----------------------
# UI
# -----------------------
st.title("🎬 Movie Recommendation System")
st.write("Keyword-based movie recommendation using ML + TMDB")

keyword = st.text_input("Enter a keyword (love, romance, thriller, ghost, etc)")

if st.button("Recommend") and keyword:
    results = recommend_by_keywords(keyword)

    cols = st.columns(5)
    for i, row in enumerate(results.itertuples()):
        poster = fetch_poster(row.title)

        with cols[i]:
            st.caption(row.title)
            if poster:
                st.image(poster, use_container_width=True)
            else:
                st.markdown(
                    f"""
                    <div style="height:300px;display:flex;
                    align-items:center;justify-content:center;
                    background:#f0f0f0;border-radius:10px;">
                    <b>{row.title}</b>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
