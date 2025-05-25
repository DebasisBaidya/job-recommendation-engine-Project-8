import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from io import BytesIO
from fpdf import FPDF

st.set_page_config(page_title="Job Role Recommender", layout="centered")

@st.cache_resource
def load_resources():
    try:
        with open("job_recommender_model.pkl", "rb") as f:
            model = pickle.load(f)
    except Exception:
        model = None
    try:
        with open("vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)
    except Exception:
        vectorizer = None
    try:
        data = pd.read_csv("job_data.csv", parse_dates=["published_date"])
    except Exception:
        data = pd.DataFrame()
    return model, vectorizer, data

model, vectorizer, data = load_resources()

if data.empty:
    st.error("❌ Job data not found or empty.")
    st.stop()

# Silent fallback for missing columns
if "processed_text" not in data.columns:
    data["processed_text"] = data["keywords"].fillna("").astype(str)

if "title" not in data.columns:
    data["title"] = data["processed_text"].apply(lambda x: " ".join(x.split()[:5]) if isinstance(x, str) else "N/A")

data["location"] = data.get("location", "Unknown")
data["experience"] = data.get("experience", pd.Series()).fillna(
    data["keywords"].str.extract(r'(Fresher|Experienced)', expand=False)
).fillna("Not Specified")

data["job_type"] = data.get("job_type", pd.Series()).fillna(
    data["keywords"].str.extract(r'(Remote|Hybrid|Freelance|On-site|Full-Time|Part-Time|Contract)', expand=False)
).fillna("Unknown")

data["country"] = data.get("country", "Unknown")

experience_levels = ["Fresher", "Experienced"]
locations = sorted(data["country"].replace("", np.nan).dropna().unique())
job_types = ["Remote", "On-site", "Hybrid", "Freelance", "Full-Time", "Part-Time", "Contract"]

st.markdown("""
    <div style='text-align: center; margin-bottom: 15px;'>
        <h1 style="font-weight: 700; color: #4B6EAF; font-size: 28px;">🔍 AI-Powered Job Role Recommender</h1>
        <p style="font-size: 14px; color: #555; margin-top: -10px;">Get top job suggestions based on your description, with filters.</p>
    </div>
""", unsafe_allow_html=True)

if "job_desc" not in st.session_state:
    st.session_state.job_desc = ""

if "keyword_results" not in st.session_state:
    st.session_state.keyword_results = None

with st.form(key="recommend_form"):
    # Updated label with example line below, styled nicely
    st.markdown("""
    <div style='text-align: center; margin-bottom: 2px;'>
        <h3 style="color: #333; font-size: 18px;">📝 Enter Job Description</h3>
    </div>
    <div style='text-align: center; margin-bottom: 12px; font-size: 13px; color: #666; font-style: italic;'>
        Example: Facebook and YouTube Ads Media Buyer for Roofing companies
    </div>
    """, unsafe_allow_html=True)

    job_desc = st.text_area(
        "", 
        placeholder="Describe the job role you're looking for...", 
        height=80, 
        value=st.session_state.job_desc
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        exp_filter = st.multiselect("👤 Experience Level(s)", experience_levels)
    with col2:
        location_filter = st.multiselect("🌍 Country/Countries", locations)
    with col3:
        job_type_filter = st.multiselect("🧑‍💻 Job Type(s)", job_types)

    btn_col1, btn_col2, btn_col3 = st.columns([1,1,1])
    with btn_col2:
        submit = st.form_submit_button("🔎 Recommend Jobs")
    with btn_col3:
        reset = st.form_submit_button("♻️ Reset All")

    if submit:
        st.session_state.job_desc = job_desc
    if reset:
        st.session_state.job_desc = ""
        st.session_state.keyword_results = None
        st.experimental_rerun()

def generate_pdf(dataframe):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=14)
    pdf.cell(0, 10, "Job Recommendations", ln=True, align="C")
    pdf.ln(5)

    pdf.set_font("Arial", size=10)
    for idx, row in dataframe.iterrows():
        pdf.set_text_color(27, 38, 49)
        pdf.cell(0, 8, f"{idx+1}. {row['title']}", ln=True)
        pdf.set_font("Arial", size=9)
        pdf.set_text_color(80, 80, 80)
        pdf.cell(0, 6, f"Location: {row['country']} | Date: {pd.to_datetime(row['published_date']).date() if pd.notna(row['published_date']) else 'N/A'}", ln=True)
        pdf.cell(0, 6, f"Experience: {row['experience']} | Type: {row['job_type']}", ln=True)
        pdf.multi_cell(0, 6, f"Keywords: {row['keywords']}")
        pdf.set_font("Arial", size=10)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(0, 6, f"Similarity: {row['Similarity (%)']}%", ln=True)
        pdf.ln(4)

    pdf_bytes = pdf.output(dest='S').encode('latin1')
    return BytesIO(pdf_bytes)

def strip_html_tags(text):
    if not isinstance(text, str):
        return text
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

def process_and_display_results():
    if not st.session_state.job_desc.strip():
        st.warning("⚠️ Please enter a job description.")
        return

    if model is None or vectorizer is None:
        st.error("⚠️ Model or vectorizer not loaded.")
        return

    user_vec = vectorizer.transform([st.session_state.job_desc])
    if user_vec.nnz == 0:
        st.warning("⚠️ Your job description is too vague or contains unfamiliar terms. Try adding more relevant keywords.")
        return

    try:
        distances, indices = model.kneighbors(user_vec, n_neighbors=10)
    except ValueError:
        st.error("⚠️ Unable to find similar jobs. Please try rephrasing your job description.")
        return

    results = data.iloc[indices[0]].copy()
    results["Similarity (%)"] = [round((1 - d) * 100, 2) for d in distances[0]]
    keyword_results = results.copy()

    if exp_filter:
        filtered = keyword_results[keyword_results["experience"].isin(exp_filter)]
        if not filtered.empty:
            keyword_results = filtered

    if job_type_filter:
        filtered = keyword_results[keyword_results["job_type"].isin(job_type_filter)]
        if not filtered.empty:
            keyword_results = filtered

    if location_filter:
        location_matched = keyword_results[keyword_results["country"].isin(location_filter)]
        if location_matched.empty:
            alt_countries = keyword_results["country"].value_counts().head(5)
            if not alt_countries.empty:
                alt_country_list = ", ".join(alt_countries.index.tolist())
                st.info(f"📍 No jobs found in selected country/countries, but available in: {alt_country_list}")
                keyword_results = keyword_results[keyword_results["country"].isin(alt_countries.index)]
        else:
            keyword_results = location_matched

    if keyword_results.empty:
        st.warning("😕 No matching jobs found.")
        st.session_state.keyword_results = None
        return

    keyword_results = keyword_results.sort_values("Similarity (%)", ascending=False).reset_index(drop=True)
    keyword_results.index += 1
    keyword_results["Rank"] = keyword_results.index

    def extract_keywords(text):
        return set(re.findall(r"\b\w{4,}\b", text.lower()))

    keywords = extract_keywords(st.session_state.job_desc)

    def highlight_keywords(text):
        for word in keywords:
            pattern = re.compile(rf"\b({re.escape(word)})\b", re.IGNORECASE)
            text = pattern.sub(r"<mark><b>\1</b></mark>", text)
        return text

    keyword_results["title"] = keyword_results["title"].fillna("N/A")
    keyword_results["title_highlighted"] = keyword_results["title"].astype(str).apply(highlight_keywords)
    keyword_results["keywords"] = keyword_results["keywords"].astype(str).apply(highlight_keywords)

    st.session_state.keyword_results = keyword_results

    st.success("✅ Top Matching Job Roles:")
    for _, row in keyword_results.iterrows():
        st.markdown(f"""
        <div style="
            padding: 8px; 
            border: 1px solid #ddd; 
            border-radius: 8px; 
            margin-bottom: 8px; 
            background-color: #f9f9f9;
            font-size: 12px;
            line-height: 1.2;
            ">
            <h4 style="color: #2c3e50; margin-bottom: 4px; font-size: 14px;">🔹 {row['title_highlighted']}</h4>
            <p style="color: #555; margin: 1px 0;">
                <strong>Location:</strong> {row['country']} &nbsp;&nbsp; 
                <strong>Date:</strong> {pd.to_datetime(row['published_date']).date() if pd.notna(row['published_date']) else 'N/A'} &nbsp;&nbsp; 
                <strong>Experience:</strong> {row['experience']} &nbsp;&nbsp;
                <strong>Type:</strong> {row['job_type']}
            </p>
            <p style="color: #666; margin: 1px 0;"><strong>Keywords:</strong> {row['keywords']}</p>
            <p style="color: #1a73e8; font-weight: 600; margin: 1px 0;">Similarity: {row['Similarity (%)']}%</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    top_countries = keyword_results["country"].value_counts().head(5)
    if not top_countries.empty:
        st.markdown("### 🌎 Top 5 Available Countries")
        fig, ax = plt.subplots(figsize=(6, 3))
        top_countries.plot(kind='bar', ax=ax, color='#4B6EAF')
        for i, (label, value) in enumerate(top_countries.items()):
            ax.text(i, value + 0.1, str(value), ha='center', fontweight='bold', fontsize=9)
        plt.title("Top 5 Available Countries", fontsize=12)
        plt.ylabel("Job Count", fontsize=10)
        plt.xticks(rotation=45, fontsize=10)
        plt.yticks(fontsize=10)
        plt.tight_layout()
        st.pyplot(fig)

    pdf_ready_df = keyword_results.copy()
    pdf_ready_df['keywords'] = pdf_ready_df['keywords'].apply(strip_html_tags)
    pdf_ready_df['title'] = pdf_ready_df['title'].apply(strip_html_tags)

    pdf_bytes = generate_pdf(pdf_ready_df)

    col1, col2, col3 = st.columns(3)
    with col2:
        st.download_button(
            label="📥 Download Recommendations as PDF",
            data=pdf_bytes,
            file_name="job_recommendations.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
        st.markdown(
            """
            <style>
            div.stDownloadButton > button:first-child {
                margin-left: auto;
                margin-right: auto;
                display: block;
                font-size: 14px;
                padding: 6px 20px;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

if st.session_state.keyword_results is not None:
    process_and_display_results()
elif 'submit' in locals() and submit:
    process_and_display_results()
