import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import re

# ---------------------- Page Setup ----------------------
st.set_page_config(page_title="Job Role Recommender", layout="centered")

# ---------------------- Load Resources ----------------------
@st.cache_resource
def load_resources():
    try:
        with open("job_recommender_model.pkl", "rb") as f:
            model = pickle.load(f)
    except:
        model = None

    try:
        with open("vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)
    except:
        vectorizer = None

    try:
        data = pd.read_csv("job_data.csv", parse_dates=["published_date"])
    except:
        data = pd.DataFrame()

    return model, vectorizer, data

model, vectorizer, data = load_resources()

# ---------------------- Error Handling ----------------------
if data.empty or 'processed_text' not in data.columns:
    st.error("❌ Job data not found or is empty or missing 'processed_text' column.")
    st.stop()

# ---------------------- UI Header ----------------------
st.markdown("<h1 style='text-align: center;'>🔍 AI-Powered Job Role Recommender</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Enter a job description and get AI-powered suggestions with similarity scores and highlights.</p>", unsafe_allow_html=True)

# ---------------------- Job Input Form ----------------------
categories = sorted(data["category"].dropna().unique())
st.markdown("<br>", unsafe_allow_html=True)

with st.form(key="recommend_form"):
    st.markdown("### 📝 Job Description")
    job_desc = st.text_area("", placeholder="Describe the job role you're looking for...", height=150)

    st.markdown("### 📂 Select Category (Optional)")
    job_category = st.selectbox("", [""] + categories)

    submit = st.form_submit_button("🔎 Recommend Jobs")

# ---------------------- Preprocessing Function ----------------------
def clean_text(text):
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove non-alphabetic
    text = re.sub(r"\s+", " ", text)         # Remove extra whitespace
    return text.lower().strip()

# ---------------------- On Submit ----------------------
if submit:
    if not job_desc.strip():
        st.warning("⚠️ Please enter a job description.")
    elif model is None or vectorizer is None:
        st.error("⚠️ Model or vectorizer not loaded.")
    else:
        # Preprocess and vectorize user input
        cleaned_input = clean_text(job_desc)
        user_vec = vectorizer.transform([cleaned_input])
        distances, indices = model.kneighbors(user_vec, n_neighbors=6)

        results = data.iloc[indices[0]].copy()
        results["Similarity (%)"] = [round((1 - d) * 100, 2) for d in distances[0]]

        # Filter category (optional)
        if job_category:
            results = results[results["category"] == job_category]

        if results.empty:
            st.warning("😕 No matching jobs found.")
        else:
            results = results.sort_values("Similarity (%)", ascending=False).reset_index(drop=True)
            results.index += 1
            results["Rank"] = results.index

            # ------------------ Keyword Highlighting ------------------
            def extract_keywords(text):
                return set(re.findall(r"\b\w{4,}\b", text.lower()))

            keywords = extract_keywords(job_desc)

            def highlight_keywords(title):
                for word in keywords:
                    title = re.sub(f"(?i)\\b({word})\\b", r"<mark><b>\1</b></mark>", title)
                return title

            results["title_highlighted"] = results["title"].apply(highlight_keywords)

            # ------------------ Display Table with Highlights ------------------
            st.success("✅ Top Matching Job Roles:")
            for i, row in results.iterrows():
                st.markdown(f"""
                <div style="padding: 10px; border: 1px solid #ddd; border-radius: 8px; margin-bottom: 10px;">
                    <h4>🔹 {row['title_highlighted']}</h4>
                    <p><strong>Company:</strong> {row['company']} &nbsp;&nbsp; 
                       <strong>Location:</strong> {row['location']} &nbsp;&nbsp; 
                       <strong>Date:</strong> {row['published_date'].date()} &nbsp;&nbsp; 
                       <strong>Category:</strong> {row['category']}</p>
                    <p><strong>Similarity Score:</strong> {row['Similarity (%)']}%</p>
                </div>
                """, unsafe_allow_html=True)

            # ------------------ Similarity Score Chart ------------------
            st.markdown("### 📊 Similarity Score Chart")
            fig, ax = plt.subplots()
            ax.bar(results["Rank"], results["Similarity (%)"], color="#4C9F70")
            ax.set_xlabel("Rank")
            ax.set_ylabel("Similarity (%)")
            ax.set_title("Top 5 Job Role Similarities")
            st.pyplot(fig)

            # ------------------ Download Button ------------------
            download_cols = ["Rank", "title", "company", "location", "published_date", "category", "Similarity (%)"]
            csv_data = results[download_cols].to_csv(index=False)
            st.download_button("📥 Download Recommendations as CSV", csv_data, "job_recommendations.csv", "text/csv")
