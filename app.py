import streamlit as st
import pandas as pd
import requests
import urllib.parse

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Movie Recommendation System",
    layout="wide"
)

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("final_movies.csv")
    return df

movies = load_data()

# -------------------------------------------------
# VECTORIZE TAGS
# -------------------------------------------------
@st.cache_data
def build_model(data):
    cv = CountVectorizer(max_features=5000, stop_words="english")
    vectors = cv.fit_transform(data["tags"]).toarray()
    similarity = cosine_similarity(vectors)
    return cv, vectors, similarity

cv, vectors, similarity = build_model(movies)

# -------------------------------------------------
# RECOMMENDER (KEYWORD BASED)
# -------------------------------------------------
def recommend_by_keywords(keyword, n=5):
    keyword_vector = cv.transform([keyword])
    scores = cosine_similarity(keyword_vector, vectors).flatten()
    top_indices = scores.argsort()[-n:][::-1]
    return movies.iloc[top_indices]["title"].tolist()

# -------------------------------------------------
# TMDB API (USING STREAMLIT SECRETS)
# -------------------------------------------------
API_KEY = st.secrets["TMDB_API_KEY"]

@st.cache_data(show_spinner=False)
def fetch_poster(movie_name):
    try:
        query = urllib.parse.quote(movie_name)
        search_url = (
            f"https://api.themoviedb.org/3/search/movie"
            f"?api_key={API_KEY}&query={query}"
        )
        data = requests.get(search_url, timeout=5).json()

        if not data.get("results"):
            return None

        poster_path = data["results"][0].get("poster_path")
        if not poster_path:
            return None

        return "https://image.tmdb.org/t/p/w500" + poster_path

    except Exception:
        return None

# -------------------------------------------------
# STREAMLIT UI
# -------------------------------------------------
st.title("🎬 Movie Recommendation System")
st.write("Keyword-based movie recommendation with TMDB posters")

keyword = st.text_input(
    "Enter a keyword (love, romance, ghost, thriller, etc)"
)

if st.button("Recommend"):
    if keyword.strip() == "":
        st.warning("Please enter a keyword")
    else:
        results = recommend_by_keywords(keyword)

        st.subheader("Recommended Movies")
        cols = st.columns(5)

        for idx, movie in enumerate(results):
            poster = fetch_poster(movie)

            with cols[idx]:
                st.caption(movie)

                if poster:
                    st.image(poster, width=200)
                else:
                    st.markdown(
                        f"""
                        <div style="
                            width:200px;
                            height:300px;
                            display:flex;
                            align-items:center;
                            justify-content:center;
                            background-color:#f0f0f0;
                            border-radius:10px;
                            text-align:center;
                            font-weight:bold;
                        ">
                            {movie}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
