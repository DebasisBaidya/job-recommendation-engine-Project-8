import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
import random
from io import BytesIO
from fpdf import FPDF

# Set Streamlit page config for centered layout and custom title
st.set_page_config(page_title="Job Role Recommender", layout="centered")

@st.cache_resource
def load_resources():
    """
    Load the pre-trained model, vectorizer, and job data from disk.
    Use try-except blocks to handle missing files gracefully.
    Cache the result to avoid reloading on every interaction.
    """
    try:
        with open("job_recommender_model.pkl", "rb") as f:
            model = pickle.load(f)
    except:
        model = None  # Model not found or failed to load

    try:
        with open("vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)
    except:
        vectorizer = None  # Vectorizer not found or failed to load

    try:
        data = pd.read_csv("job_data.csv", parse_dates=["published_date"])
    except:
        data = pd.DataFrame()  # Empty DataFrame if file missing or error

    return model, vectorizer, data

# Load model, vectorizer and data at app start
model, vectorizer, data = load_resources()

# Stop execution if job data is missing or empty
if data.empty:
    st.error("❌ Job data not found or is empty.")
    st.stop()

# Ensure 'processed_text' column exists, fallback to 'keywords' if missing
if "processed_text" not in data.columns:
    st.warning("🛠 'processed_text' column missing. Using 'keywords' as fallback.")
    data["processed_text"] = data["keywords"].fillna("").astype(str)

# Ensure 'title' column exists, fallback to first 5 words of processed_text if missing
if "title" not in data.columns:
    st.warning("🛠 'title' column missing. Using first 5 words of processed_text as fallback.")
    data["title"] = data["processed_text"].apply(lambda x: " ".join(x.split()[:5]) if isinstance(x, str) else "N/A")

# Fill missing columns with default values or extract from keywords where possible
data["location"] = data.get("location", "Unknown")
data["experience"] = data.get("experience", pd.Series()).fillna(
    data["keywords"].str.extract(r'(Fresher|Experienced)', expand=False)
).fillna("Not Specified")

data["job_type"] = data.get("job_type", pd.Series()).fillna(
    data["keywords"].str.extract(r'(Remote|Hybrid|Freelance|On-site|Full-Time|Part-Time|Contract)', expand=False)
).fillna("Unknown")

data["country"] = data.get("country", "Unknown")

# Define filter options for user selection
experience_levels = ["Fresher", "Experienced"]
locations = sorted(data["country"].replace("", np.nan).dropna().unique())
job_types = ["Remote", "On-site", "Hybrid", "Freelance", "Full-Time", "Part-Time", "Contract"]

# Display main header and description with centered alignment and styling
st.markdown("""
    <div style='text-align: center; margin-bottom: 20px;'>
        <h1 style="font-weight: 700; color: #4B6EAF;">🔍 AI-Powered Job Role Recommender</h1>
        <p style="font-size: 18px; color: #555;">Get top job suggestions based on your description, with experience, location, and job type filters.</p>
    </div>
""", unsafe_allow_html=True)

# Create a form for user input to improve UX and handle submission cleanly
with st.form(key="recommend_form"):
    # Section header for job description input
    st.markdown("""
    <div style='text-align: center; margin-bottom: 10px;'>
        <h3 style="color: #333;">📝 Job Description</h3>
    </div>
    """, unsafe_allow_html=True)

    # Text area for user to enter job description
    job_desc = st.text_area("", placeholder="Describe the job role you're looking for...", height=100)

    # Three columns for optional filters: experience, location, job type
    col1, col2, col3 = st.columns(3)
    with col1:
        exp_filter = st.multiselect("👤 Experience Level(s) (Optional)", experience_levels)
    with col2:
        location_filter = st.multiselect("🌍 Country/Countries (Optional)", locations)
    with col3:
        job_type_filter = st.multiselect("🧑‍💻 Job Type(s) (Optional)", job_types)

    # Submit button for the form
    submit = st.form_submit_button("🔎 Recommend Jobs")

def generate_pdf(dataframe):
    """
    Generate a PDF file from the job recommendations DataFrame.
    Use FPDF to create a clean, readable PDF with job details.
    Returns a BytesIO stream for download.
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=14)
    pdf.cell(0, 10, "Job Recommendations", ln=True, align="C")
    pdf.ln(5)

    pdf.set_font("Arial", size=10)
    for idx, row in dataframe.iterrows():
        # Job title in dark blue
        pdf.set_text_color(27, 38, 49)
        pdf.cell(0, 8, f"{idx+1}. {row['title']}", ln=True)

        # Job details in smaller gray text
        pdf.set_font("Arial", size=9)
        pdf.set_text_color(80, 80, 80)
        pdf.cell(0, 6, f"Location: {row['country']} | Date: {pd.to_datetime(row['published_date']).date() if pd.notna(row['published_date']) else 'N/A'}", ln=True)
        pdf.cell(0, 6, f"Experience: {row['experience']} | Type: {row['job_type']}", ln=True)

        # Keywords may wrap into multiple lines
        pdf.multi_cell(0, 6, f"Keywords: {row['keywords']}")

        # Similarity score in black
        pdf.set_font("Arial", size=10)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(0, 6, f"Similarity: {row['Similarity (%)']}%", ln=True)
        pdf.ln(4)

    # Get PDF as bytes string and wrap in BytesIO for Streamlit download
    pdf_bytes = pdf.output(dest='S').encode('latin1')
    return BytesIO(pdf_bytes)

# Process form submission
if submit:
    # Validate job description input
    if not job_desc.strip():
        st.warning("⚠️ Please enter a job description.")
    # Check if model and vectorizer loaded correctly
    elif model is None or vectorizer is None:
        st.error("⚠️ Model or vectorizer not loaded.")
    else:
        # Transform user input to vector space
        user_vec = vectorizer.transform([job_desc])
        if user_vec.nnz == 0:
            # Warn if input is too vague or unknown to vectorizer
            st.warning("⚠️ Your job description is too vague or contains unfamiliar terms. Try adding more relevant keywords.")
            st.stop()

        try:
            # Find top 10 nearest jobs using the model
            distances, indices = model.kneighbors(user_vec, n_neighbors=10)
        except ValueError as e:
            # Handle errors during nearest neighbor search
            st.error("⚠️ Unable to find similar jobs. Please try rephrasing your job description.")
            st.stop()

        # Extract matching job rows and calculate similarity percentage
        results = data.iloc[indices[0]].copy()
        results["Similarity (%)"] = [round((1 - d) * 100, 2) for d in distances[0]]
        keyword_results = results.copy()

        # Apply experience level filter if selected
        if exp_filter:
            filtered = keyword_results[keyword_results["experience"].isin(exp_filter)]
            if not filtered.empty:
                keyword_results = filtered

        # Apply job type filter if selected
        if job_type_filter:
            filtered = keyword_results[keyword_results["job_type"].isin(job_type_filter)]
            if not filtered.empty:
                keyword_results = filtered

        # Apply location filter if selected, with fallback info if no matches
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

        # Show warning if no jobs found after filtering
        if keyword_results.empty:
            st.warning("😕 No matching jobs found.")
        else:
            # Sort results by similarity descending and reset index for ranking
            keyword_results = keyword_results.sort_values("Similarity (%)", ascending=False).reset_index(drop=True)
            keyword_results.index += 1
            keyword_results["Rank"] = keyword_results.index

            # Extract keywords from user input to highlight in results
            def extract_keywords(text):
                return set(re.findall(r"\b\w{4,}\b", text.lower()))

            keywords = extract_keywords(job_desc)

            # Highlight user keywords in job title and keywords fields using HTML mark tags
            def highlight_keywords(text):
                for word in keywords:
                    pattern = re.compile(rf"\b({re.escape(word)})\b", re.IGNORECASE)
                    text = pattern.sub(r"<mark><b>\1</b></mark>", text)
                return text

            keyword_results["title"] = keyword_results["title"].fillna("N/A")
            keyword_results["title_highlighted"] = keyword_results["title"].astype(str).apply(highlight_keywords)
            keyword_results["keywords"] = keyword_results["keywords"].astype(str).apply(highlight_keywords)

            # Display the top matching jobs with styled containers
            st.success("✅ Top Matching Job Roles:")
            for _, row in keyword_results.iterrows():
                st.markdown(f"""
                <div style="padding: 15px; border: 1px solid #ddd; border-radius: 10px; margin-bottom: 15px; background-color: #f9f9f9;">
                    <h4 style="color: #2c3e50;">🔹 {row['title_highlighted']}</h4>
                    <p style="color: #555;"><strong>Location:</strong> {row['country']} &nbsp;&nbsp; 
                       <strong>Date:</strong> {pd.to_datetime(row['published_date']).date() if pd.notna(row['published_date']) else 'N/A'} &nbsp;&nbsp; 
                       <strong>Experience:</strong> {row['experience']} &nbsp;&nbsp;
                       <strong>Type:</strong> {row['job_type']}</p>
                    <p style="color: #666;"><strong>Keywords:</strong> {row['keywords']}</p>
                </div>
                """, unsafe_allow_html=True)

            # Show a bar chart of top 5 countries with job counts
            st.markdown("---")
            top_countries = keyword_results["country"].value_counts().head(5)
            if not top_countries.empty:
                st.markdown("### 🌎 Top 5 Available Countries")
                fig, ax = plt.subplots()
                top_countries.plot(kind='bar', ax=ax, color='#4B6EAF')
                for i, (label, value) in enumerate(top_countries.items()):
                    ax.text(i, value + 0.1, str(value), ha='center', fontweight='bold')
                plt.title("Top 5 Available Countries")
                plt.ylabel("Job Count")
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)

            # Generate PDF from filtered results and provide a centered download button
            pdf_bytes = generate_pdf(keyword_results)

            # Use Streamlit columns to center the download button horizontally
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.download_button(
                    label="📥 Download Recommendations as PDF",
                    data=pdf_bytes,
                    file_name="job_recommendations.pdf",
                    mime="application/pdf"
                )
