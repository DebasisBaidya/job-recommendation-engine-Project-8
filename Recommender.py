import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
import random

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

if "processed_text" not in data.columns:
    st.warning("🛠 'processed_text' column missing. Using 'keywords' as fallback.")
    data["processed_text"] = data["keywords"].fillna("").astype(str)

company_names = ["TechNova", "DataX", "InnoCore", "SoftWave", "BrightPath", "NextEdge"]
if "company" not in data.columns or data["company"].isna().all():
    data["company"] = [random.choice(company_names) for _ in range(len(data))]
else:
    data["company"] = data["company"].fillna(random.choice(company_names))

if "location" not in data.columns:
    data["location"] = "Unknown"

if "experience" not in data.columns:
    data["experience"] = data["keywords"].str.extract(r'(Fresher|Experienced)', expand=False).fillna("Not Specified")
else:
    data["experience"] = data["experience"].fillna(data["keywords"].str.extract(r'(Fresher|Experienced)', expand=False)).fillna("Not Specified")

if "job_type" not in data.columns:
    data["job_type"] = data["keywords"].str.extract(r'(Remote|Hybrid|Freelance|On-site|Full-Time|Part-Time|Contract)', expand=False).fillna("Unknown")
else:
    data["job_type"] = data["job_type"].fillna(data["keywords"].str.extract(r'(Remote|Hybrid|Freelance|On-site|Full-Time|Part-Time|Contract)', expand=False)).fillna("Unknown")

if "country" not in data.columns:
    data["country"] = "Unknown"

experience_levels = ["Fresher", "Experienced"]
locations = sorted(data["country"].replace("", np.nan).dropna().unique())
job_types = ["Remote", "On-site", "Hybrid", "Freelance", "Full-Time", "Part-Time", "Contract"]

# ---------------------- UI Header ----------------------
st.markdown("<h1 style='text-align: center;'>🔍 AI-Powered Job Role Recommender</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Get top job suggestions based on your description, with experience, location, and job type filters.</p>", unsafe_allow_html=True)

# ---------------------- Input Form ----------------------
with st.form(key="recommend_form"):
    st.markdown("### 📝 Job Description")
    job_desc = st.text_area("", placeholder="Describe the job role you're looking for...", height=150)

    st.markdown("### 👤 Select Experience Level(s) (Optional)")
    exp_filter = st.multiselect("", experience_levels)

    st.markdown("### 🌍 Select Country/Countries (Optional)")
    location_filter = st.multiselect("", locations)

    st.markdown("### 🧑‍💻 Select Job Type(s) (Optional)")
    job_type_filter = st.multiselect("", job_types)

    submit = st.form_submit_button("🔎 Recommend Jobs")

# ---------------------- On Submit ----------------------
if submit:
    if not job_desc.strip():
        st.warning("⚠️ Please enter a job description.")
    elif model is None or vectorizer is None:
        st.error("⚠️ Model or vectorizer not loaded.")
    else:
        user_vec = vectorizer.transform([job_desc])
        distances, indices = model.kneighbors(user_vec, n_neighbors=10)

        if len(indices) == 0 or len(indices[0]) == 0:
            st.warning("😕 No matching jobs found.")
        else:
            results = data.iloc[indices[0]].copy()
            results["Similarity (%)"] = [round((1 - d) * 100, 2) for d in distances[0]]
            keyword_results = results.copy()

            if exp_filter:
                filtered = keyword_results[keyword_results["experience"].isin(exp_filter)]
                if not filtered.empty:
                    keyword_results = filtered

            if job_type_filter and "job_type" in keyword_results.columns:
                filtered = keyword_results[keyword_results["job_type"].isin(job_type_filter)]
                if not filtered.empty:
                    keyword_results = filtered

            location_displayed = False
            if location_filter:
                location_matched = keyword_results[keyword_results["country"].isin(location_filter)]
                if location_matched.empty:
                    alt_countries = keyword_results["country"].value_counts().head(5)
                    if not alt_countries.empty:
                        alt_country_list = ", ".join(alt_countries.index.tolist())
                        st.info(f"📍 No jobs found in selected country/countries, but available in: {alt_country_list}")
                        fig, ax = plt.subplots()
                        alt_countries.plot(kind='bar', ax=ax, color='skyblue')
                        for i, (label, value) in enumerate(alt_countries.items()):
                            ax.text(i, value + 0.1, str(value), ha='center', fontweight='bold')
                        plt.title("Top 5 Available Countries")
                        plt.ylabel("Job Count")
                        st.pyplot(fig)
                        keyword_results = keyword_results[keyword_results["country"].isin(alt_countries.index)]
                        location_displayed = True
                else:
                    keyword_results = location_matched

            if keyword_results.empty:
                st.warning("😕 No matching jobs found.")
            else:
                keyword_results = keyword_results.sort_values("Similarity (%)", ascending=False).reset_index(drop=True)
                keyword_results.index += 1
                keyword_results["Rank"] = keyword_results.index

                def extract_keywords(text):
                    return set(re.findall(r"\b\w{4,}\b", text.lower()))

                keywords = extract_keywords(job_desc)

                def highlight_keywords(text):
                    for word in keywords:
                        pattern = re.compile(rf"\b({re.escape(word)})\b", re.IGNORECASE)
                        text = pattern.sub(r"<mark><b>\1</b></mark>", text)
                    return text

                if isinstance(keyword_results["processed_text"], pd.Series):
                    keyword_results["title_highlighted"] = keyword_results["processed_text"].astype(str).apply(highlight_keywords)
                if isinstance(keyword_results["company"], pd.Series):
                    keyword_results["company"] = keyword_results["company"].astype(str).apply(highlight_keywords)
                if isinstance(keyword_results["keywords"], pd.Series):
                    keyword_results["keywords"] = keyword_results["keywords"].astype(str).apply(highlight_keywords)

                st.success("✅ Top Matching Job Roles:")
                for _, row in keyword_results.iterrows():
                    st.markdown(f"""
                    <div style="padding: 10px; border: 1px solid #ddd; border-radius: 8px; margin-bottom: 10px;">
                        <h4>🔹 {row['title_highlighted']}</h4>
                        <p><strong>Company:</strong> {row['company']} &nbsp;&nbsp; 
                           <strong>Location:</strong> {row['country']} &nbsp;&nbsp; 
                           <strong>Date:</strong> {pd.to_datetime(row['published_date']).date() if pd.notna(row['published_date']) else 'N/A'} &nbsp;&nbsp; 
                           <strong>Experience:</strong> {row['experience']} &nbsp;&nbsp;
                           <strong>Type:</strong> {row['job_type']}</p>
                        <p><strong>Keywords:</strong> {row['keywords']}</p>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("---")
                top_countries = keyword_results["country"].value_counts().head(5)
                if not top_countries.empty:
                    st.markdown("### 🌎 Top 5 Available Countries")
                    fig, ax = plt.subplots()
                    top_countries.plot(kind='bar', ax=ax, color='salmon')
                    for i, (label, value) in enumerate(top_countries.items()):
                        ax.text(i, value + 0.1, str(value), ha='center', fontweight='bold')
                    plt.title("Top 5 Available Countries")
                    plt.ylabel("Job Count")
                    st.pyplot(fig)

                download_cols = ["Rank", "processed_text", "company", "country", "published_date", "experience", "job_type", "Similarity (%)"]
                csv_data = keyword_results[download_cols].to_csv(index=False)
                st.download_button("📥 Download Recommendations as CSV", csv_data, "job_recommendations.csv", "text/csv")
