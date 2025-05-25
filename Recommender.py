import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer

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

# ---------------------- Preprocess Data ----------------------
if data.empty:
    st.error("❌ Job data not found or is empty.")
    st.stop()

# Ensure necessary columns
if "processed_text" not in data.columns:
    st.warning("🛠 'processed_text' column missing. Using 'keywords' as fallback.")
    data["processed_text"] = data["keywords"].fillna("").astype(str)

if "company" not in data.columns:
    data["company"] = [f"Company {i+1}" for i in range(len(data))]

if "location" not in data.columns:
    data["location"] = "Unknown"

if "category" not in data.columns:
    data["category"] = "General"

if "job_type" not in data.columns:
    data["job_type"] = data["keywords"].str.extract(r'(Remote|Hybrid|Freelance|On-site|Full-Time|Part-Time|Contract)', expand=False).fillna("Unknown")

# Get unique dropdown options
categories = sorted(data["category"].dropna().unique())
locations = sorted(data["country"].dropna().unique())
job_types = sorted(data["job_type"].dropna().unique())

# ---------------------- UI Header ----------------------
st.markdown("<h1 style='text-align: center;'>🔍 AI-Powered Job Role Recommender</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Get top job suggestions based on your description, with category, location, and job type filters.</p>", unsafe_allow_html=True)

# ---------------------- Input Form ----------------------
with st.form(key="recommend_form"):
    st.markdown("### 📝 Job Description")
    job_desc = st.text_area("", placeholder="Describe the job role you're looking for...", height=150)

    st.markdown("### 📂 Select Category (Optional)")
    job_category = st.selectbox("", [""] + categories)

    st.markdown("### 🌍 Select Location (Optional)")
    location_filter = st.selectbox("", [""] + locations)

    st.markdown("### 💼 Select Job Type (Optional)")
    job_type_filter = st.selectbox("", [""] + job_types)

    submit = st.form_submit_button("🔎 Recommend Jobs")

# ---------------------- On Submit ----------------------
if submit:
    if not job_desc.strip():
        st.warning("⚠️ Please enter a job description.")
    elif model is None or vectorizer is None:
        st.error("⚠️ Model or vectorizer not loaded.")
    else:
        # Vectorize user input
        user_vec = vectorizer.transform([job_desc])
        distances, indices = model.kneighbors(user_vec, n_neighbors=10)

        # Prepare results
        results = data.iloc[indices[0]].copy()
        results["Similarity (%)"] = [round((1 - d) * 100, 2) for d in distances[0]]

        # Apply filters (if selected)
        if job_category:
            results = results[results["category"] == job_category]

        original_count = len(results)

        if location_filter:
            filtered = results[results["country"] == location_filter]
            if filtered.empty:
                st.info(f"📍 No exact match for selected location '{location_filter}'. Showing results from other countries.")
            else:
                results = filtered

        if job_type_filter:
            results = results[results["job_type"] == job_type_filter]

        # Show results
        if results.empty:
            st.warning("😕 No matching jobs found.")
        else:
            results = results.sort_values("Similarity (%)", ascending=False).reset_index(drop=True)
            results.index += 1
            results["Rank"] = results.index

            # Keyword Highlighting
            def extract_keywords(text):
                return set(re.findall(r"\b\w{4,}\b", text.lower()))

            keywords = extract_keywords(job_desc)

            def highlight_keywords(title):
                for word in keywords:
                    title = re.sub(f"(?i)\\b({word})\\b", r"<mark><b>\\1</b></mark>", title)
                return title

            results["title_highlighted"] = results["processed_text"].apply(highlight_keywords)

            st.success("✅ Top Matching Job Roles:")
            for _, row in results.iterrows():
                st.markdown(f"""
                <div style="padding: 10px; border: 1px solid #ddd; border-radius: 8px; margin-bottom: 10px;">
                    <h4>🔹 {row['title_highlighted']}</h4>
                    <p><strong>Company:</strong> {row['company']} &nbsp;&nbsp; 
                       <strong>Location:</strong> {row['country']} &nbsp;&nbsp; 
                       <strong>Date:</strong> {row['published_date'].date() if pd.notna(row['published_date']) else 'N/A'} &nbsp;&nbsp; 
                       <strong>Category:</strong> {row['category']} &nbsp;&nbsp;
                       <strong>Type:</strong> {row['job_type']}</p>
                    <p><strong>Similarity Score:</strong> {row['Similarity (%)']}%</p>
                </div>
                """, unsafe_allow_html=True)

            # Similarity Score Chart
            st.markdown("### 📊 Similarity Score Chart")
            fig, ax = plt.subplots()
            ax.bar(results["Rank"], results["Similarity (%)"], color="#4C9F70")
            ax.set_xlabel("Rank")
            ax.set_ylabel("Similarity (%)")
            ax.set_title("Top Job Similarities")
            st.pyplot(fig)

            # Job Count by Country Chart
            st.markdown("### 🌍 Top 5 Countries by Available Jobs")
            top_countries = data['country'].value_counts().nlargest(5)
            fig2, ax2 = plt.subplots()
            ax2.bar(top_countries.index, top_countries.values, color="#F4A261")
            ax2.set_ylabel("Number of Jobs")
            ax2.set_title("Top 5 Countries with Most Jobs")
            st.pyplot(fig2)

            # Download button
            download_cols = ["Rank", "processed_text", "company", "country", "published_date", "category", "job_type", "Similarity (%)"]
            csv_data = results[download_cols].to_csv(index=False)
            st.download_button("📥 Download Recommendations as CSV", csv_data, "job_recommendations.csv", "text/csv")
