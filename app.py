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
st.set_page_config(
    page_title="Movie Recommendation System",
    layout="wide"
)

# -----------------------
# LOAD DATA
# -----------------------
@st.cache_data
def load_data():
    return pd.read_csv("final_movies.csv")

movies = load_data()

# -----------------------
# VECTORIZE
# -----------------------
@st.cache_data
def build_model():
    cv = CountVectorizer(max_features=5000, stop_words="english")
    vectors = cv.fit_transform(movies["tags"]).toarray()
    similarity = cosine_similarity(vectors)
    return cv, vectors, similarity

cv, vectors, similarity = build_model()

# -----------------------
# RECOMMENDER
# -----------------------
def recommend_by_keywords(keyword):
    keyword_vector = cv.transform([keyword])
    scores = cosine_similarity(keyword_vector, vectors)
    top_indices = scores.flatten().argsort()[-5:][::-1]
    return movies.iloc[top_indices][["movie_id", "title"]]

# -----------------------
# TMDB API
# -----------------------
API_KEY = "b7bb86b044a3125c449ccd03a5d51b32"

@st.cache_data(show_spinner=False)
def fetch_poster(title):
    try:
        query = urllib.parse.quote(title)
        search_url = f"https://api.themoviedb.org/3/search/movie?api_key={API_KEY}&query={query}"
        res = requests.get(search_url, timeout=5).json()

        if not res["results"]:
            return None

        poster_path = res["results"][0].get("poster_path")
        if not poster_path:
            return None

        img_url = "https://image.tmdb.org/t/p/w500" + poster_path
        img = requests.get(img_url, timeout=5)
        return BytesIO(img.content)
    except:
        return None

# -----------------------
# UI
# -----------------------
st.title("🎬 Movie Recommendation System")
st.write("Keyword-based movie recommendation with TMDB posters")

keyword = st.text_input("Enter a keyword (love, romance, ghost, thriller, etc)")

if st.button("Recommend"):
    results = recommend_by_keywords(keyword)

    st.subheader("Recommended Movies")
    cols = st.columns(5)

    for idx, row in enumerate(results.itertuples()):
        poster = fetch_poster(row.title)

        with cols[idx]:
            st.caption(row.title)
            if poster:
                st.image(poster, width=200)
            else:
                st.markdown(
                    f"""
                    <div style="width:200px;height:300px;
                    display:flex;align-items:center;justify-content:center;
                    background-color:#f0f0f0;border-radius:10px;
                    font-weight:bold;text-align:center;">
                    No Poster
                    </div>
                    """,
                    unsafe_allow_html=True
                )
