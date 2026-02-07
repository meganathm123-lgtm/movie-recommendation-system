🎬 Movie Recommendation System (Streamlit + TMDB)

            A keyword-based Movie Recommendation System built using Natural Language Processing (NLP) techniques and deployed as a web application using Streamlit.
           The app recommends movies based on user-entered keywords (like love, thriller, ghost, romance) and displays real-time posters fetched from TMDB API.

🔍 Problem Statement

            With thousands of movies available, users often struggle to find content that matches their interests.
            This project solves that problem by recommending movies based on keywords extracted from genres and metadata, helping users discover relevant movies quickly.

💡 Solution Overview

Uses content-based filtering

Extracts and processes movie genres & keywords

Converts text into vectors using CountVectorizer

Measures similarity using Cosine Similarity

Fetches movie posters dynamically using TMDB API

Deployed securely using Streamlit Cloud

🛠️ Tech Stack

Category	                       Tools
Programming Language	           Python
Web Framework	           Streamlit
NLP	                       CountVectorizer
Similarity Measure                 Cosine Similarity
Data Processing	           Pandas, NumPy

API	                       TMDB API
Deployment        	           Streamlit Cloud
Version Control	           GitHub

📂 Project Structure

movie-recommendation-system/
│
├── app.py                  # Main Streamlit app
├── final_movies.csv        # Preprocessed dataset (movie_id, title, tags)
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation

⚙️ How It Works

Data Preprocessing

Merge movie and metadata datasets

Extract genres & keywords

Combine them into a single tags column

Vectorization

Convert text data into vectors using CountVectorizer

Similarity Calculation

Use cosine_similarity to find movies closest to the user keyword

Recommendation

Top 5 most similar movies are selected

Poster Fetching

Movie posters are fetched dynamically from TMDB using API calls

🚀 Live Demo

🔗 Live App: (Paste your Streamlit Cloud URL here)

🔐 API Key Security

TMDB API key is stored securely using Streamlit Secrets

The key is not hardcoded in the source code

Accessed safely using:

API_KEY = st.secrets["TMDB_API_KEY"]

📦 Installation & Local Run

1️⃣ Clone the repository
git clone https://github.com/your-username/movie-recommendation-system.git
cd movie-recommendation-system

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Add TMDB API key

Create a .streamlit/secrets.toml file:

TMDB_API_KEY = "your_api_key_here"

4️⃣ Run the app
streamlit run app.py

📈 Future Improvements

Add movie overview & rating

Add genre-based filtering

Improve recommendation accuracy using TF-IDF

Add user-based collaborative filtering

Improve UI with animations

👨‍💻 Author
Meganath M
CSE (AI & ML) Student
Movie Recommendation System using NLP & Streamlit
