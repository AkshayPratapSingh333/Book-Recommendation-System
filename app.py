import os
import sys
import pickle
import streamlit as st
import numpy as np

from book_recommendation.logger.log import logging
from book_recommendation.exception.exception_handler import AppException
from book_recommendation.config.configuration import AppConfiguration
from book_recommendation.pipeline.training_pipeline import TrainingPipeline


PLACEHOLDER_COVER = "https://placehold.co/200x300?text=No+Cover"


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Book Recommender",
    page_icon="📚",
    layout="wide",
)

st.markdown(
    """
    <style>
    .book-card {
        background: var(--background-color, #ffffff);
        border: 1px solid rgba(49, 51, 63, 0.1);
        border-radius: 12px;
        padding: 0.75rem;
        text-align: center;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
        height: 100%;
    }
    .book-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.12);
    }
    .book-card img {
        border-radius: 8px;
        margin-bottom: 0.5rem;
    }
    .book-title {
        font-weight: 600;
        font-size: 0.9rem;
        margin-bottom: 0.25rem;
    }
    .book-score {
        font-size: 0.8rem;
        color: #6c757d;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Cached loaders - artifacts are only read from disk once per session
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


class Recommendation:
    def __init__(self, app_config=AppConfiguration()):
        try:
            self.recommendation_config = app_config.get_recommendation_config()
        except Exception as e:
            raise AppException(e, sys) from e

    def fetch_poster(self, suggestion):
        try:
            book_name = []
            ids_index = []
            poster_url = []

            book_pivot = load_pickle(self.recommendation_config.book_pivot_serialized_objects)
            final_rating = load_pickle(self.recommendation_config.final_rating_serialized_objects)

            for book_id in suggestion:
                book_name.append(book_pivot.index[book_id])

            for name in book_name[0]:
                matches = np.where(final_rating["title"] == name)[0]
                ids_index.append(matches[0] if len(matches) else None)

            for idx in ids_index:
                if idx is None:
                    poster_url.append(PLACEHOLDER_COVER)
                else:
                    url = final_rating.iloc[idx]["image_url"]
                    poster_url.append(url if isinstance(url, str) and url.strip() else PLACEHOLDER_COVER)

            return poster_url

        except Exception as e:
            raise AppException(e, sys) from e

    def recommend_book(self, book_name, n_recommendations=5):
        try:
            books_list = []
            model = load_pickle(self.recommendation_config.trained_model_path)
            book_pivot = load_pickle(self.recommendation_config.book_pivot_serialized_objects)

            book_id = np.where(book_pivot.index == book_name)[0][0]
            distance, suggestion = model.kneighbors(
                book_pivot.iloc[book_id, :].values.reshape(1, -1),
                n_neighbors=n_recommendations + 1,
            )

            poster_url = self.fetch_poster(suggestion)

            for i in range(len(suggestion)):
                books = book_pivot.index[suggestion[i]]
                for j in books:
                    books_list.append(j)

            return books_list, poster_url, distance[0]

        except Exception as e:
            raise AppException(e, sys) from e

    def train_engine(self):
        try:
            obj = TrainingPipeline()
            obj.start_training_pipeline()
            st.success("Training completed!")
            logging.info("Training completed successfully!")
        except Exception as e:
            raise AppException(e, sys) from e

    def recommendations_engine(self, selected_book, n_recommendations=5):
        recommended_books, poster_url, distances = self.recommend_book(selected_book, n_recommendations)

        # index 0 is the selected book itself, so skip it
        results = list(zip(recommended_books[1:], poster_url[1:], distances[1:]))

        max_dist = max(distances[1:]) if len(distances) > 1 else 1
        cols = st.columns(len(results))

        for col, (title, poster, dist) in zip(cols, results):
            similarity = max(0.0, 1 - (dist / max_dist)) if max_dist else 0.0
            with col:
                st.markdown(
                    f"""
                    <div class="book-card">
                        <img src="{poster}" style="width:100%;" />
                        <div class="book-title">{title}</div>
                        <div class="book-score">Match: {similarity:.0%}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    st.title("📚 Book Recommender")
    st.caption("A collaborative-filtering based recommendation system. Pick a book you've enjoyed, and we'll suggest similar ones.")

    try:
        obj = Recommendation()
    except AppException as e:
        st.error("Could not load app configuration. Check the logs for details.")
        logging.error(e)
        st.stop()

    with st.sidebar:
        st.header("⚙️ Admin")
        st.write("Retrain the recommender on the latest dataset.")
        if st.button("Train Recommender System"):
            with st.spinner("Training model... this can take a while"):
                try:
                    obj.train_engine()
                except AppException as e:
                    st.error("Training failed. Check the logs for details.")
                    logging.error(e)

    try:
        book_names = load_pickle(obj.recommendation_config.book_name_serialized_objects)
    except AppException as e:
        st.error("Could not load the book list. Try training the recommender first.")
        logging.error(e)
        st.stop()

    col1, col2 = st.columns([3, 1])
    with col1:
        selected_book = st.selectbox("Type or select a book from the dropdown", book_names)
    with col2:
        n_recommendations = st.slider("Recommendations", min_value=3, max_value=10, value=5)

    if st.button("Show Recommendation", type="primary"):
        with st.spinner("Finding similar books..."):
            try:
                obj.recommendations_engine(selected_book, n_recommendations)
            except AppException as e:
                st.error("Sorry, something went wrong while generating recommendations.")
                logging.error(e)