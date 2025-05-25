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
if data.empty or model is None or vectorizer is None:
    st.error("❌ Required files (model/vectorizer/data) not found or invalid.")
    st.stop()

if "processed_text" not in data.columns:
    st.error("❌ Column 'processed_text' missing from data. Please ensure it's included in 'job_data.csv'.")
    st.stop()

# ---------------------- UI Header ----------------------
st.markdown("<h1 style='text-align: center;'>🔍 AI-Powered Job Role Recommender</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Enter a job description and get AI-powered suggestions with similarity scores.</p>", unsafe_allow_html=True)

# ---------------------- Dropdown Filters ----------------------
st.markdown("## 🎛️ Optional Filters")

all_companies = sorted(data["company"].dropna().unique().tolist())
company_input = st.selectbox("Select a Company (or type a new one)", options=[""] + all_companies)

all_countries = sorted(data["country"].dropna().unique().tolist())
location_input = st.selectbox("Select a Location", options=[""] + all_countries)

categories = sorted(data["category"].dropna().unique())
category_input = st.selectbox("Select a Category", options=[""] + categories)

# ---------------------- Job Description Form ----------------------
st.markdown("## 📝 Job Description")

with st.form("recommend_form"):
    job_desc = st.text_area("Describe the job role you're looking for", height=150)
    submitted = st.form_submit_button("🔎 Recommend Jobs")

# ---------------------- Recommendation Logic ----------------------
if submitted:
    if not job_desc.strip():
        st.warning("⚠️ Please enter a job description.")
    else:
        user_vec = vectorizer.transform([job_desc])
        distances, indices = model.kneighbors(user_vec, n_neighbors=10)

        results = data.iloc[indices[0]].copy()
        results["Similarity (%)"] = [round((1 - d) * 100, 2) for d in distances[0]]

        # Apply dropdown filters
        if company_input:
            results = results[results["company"].str.lower() == company_input.lower()]
        if location_input:
            results = results[results["country"].str.lower() == location_input.lower()]
        if category_input:
            results = results[results["category"] == category_input]

        if results.empty:
            st.warning("😕 No matching jobs found for the given filters.")
        else:
            results = results.sort_values("Similarity (%)", ascending=False).reset_index(drop=True)
            results.index += 1
            results["Rank"] = results.index

            # Highlight keywords from description
            def extract_keywords(text):
                return set(re.findall(r"\b\w{4,}\b", text.lower()))

            keywords = extract_keywords(job_desc)

            def highlight_keywords(title):
                for word in keywords:
                    title = re.sub(f"(?i)\\b({word})\\b", r"<mark><b>\1</b></mark>", title)
                return title

            results["title_highlighted"] = results["title"].apply(highlight_keywords)

            # ------------------ Display Results ------------------
            st.success("✅ Top Matching Job Roles:")
            for _, row in results.iterrows():
                st.markdown(f"""
                <div style="padding: 10px; border: 1px solid #ddd; border-radius: 8px; margin-bottom: 10px;">
                    <h4>🔹 {row['title_highlighted']}</h4>
                    <p><strong>Company:</strong> {row['company']} &nbsp;&nbsp; 
                       <strong>Location:</strong> {row['country']} &nbsp;&nbsp; 
                       <strong>Date:</strong> {row['published_date'].date()} &nbsp;&nbsp; 
                       <strong>Category:</strong> {row['category']}</p>
                    <p><strong>Similarity Score:</strong> {row['Similarity (%)']}%</p>
                </div>
                """, unsafe_allow_html=True)

            # ------------------ Similarity Chart ------------------
            st.markdown("### 📊 Similarity Score Chart")
            fig, ax = plt.subplots()
            ax.bar(results["Rank"], results["Similarity (%)"], color="#4C9F70")
            ax.set_xlabel("Rank")
            ax.set_ylabel("Similarity (%)")
            ax.set_title("Top Job Role Similarities")
            st.pyplot(fig)

            # ------------------ CSV Download ------------------
            download_cols = ["Rank", "title", "company", "country", "published_date", "category", "Similarity (%)"]
            csv_data = results[download_cols].to_csv(index=False)
            st.download_button("📥 Download Recommendations as CSV", csv_data, "job_recommendations.csv", "text/csv")
