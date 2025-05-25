import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity

# Load necessary files
@st.cache_resource
def load_model_and_data():
    model = joblib.load("models/tfidf_vectorizer.pkl")
    df = pd.read_csv("data/job_data.csv")
    if 'processed_text' not in df.columns:
        df['processed_text'] = df['title'].fillna('') + ' ' + df['category'].fillna('') + ' ' + df['keywords'].fillna('')
    return model, df

tfidf_vectorizer, jobs_df = load_model_and_data()

# Country list
locations = sorted([
    'Other', 'United States', 'India', 'Portugal', 'Germany', 'Canada', 'Singapore', 'United Kingdom',
    'Denmark', 'Malaysia', 'Bangladesh', 'Saudi Arabia', 'Australia', 'Ukraine', 'Pakistan', 'Nigeria',
    'Peru', 'Costa Rica', 'Switzerland', 'France', 'China', 'Israel', 'Hong Kong', 'Serbia', 'Bahrain',
    'Thailand', 'Spain', 'Croatia', 'Luxembourg', 'Kenya', 'Finland', 'Bulgaria', 'Georgia', 'New Zealand',
    'Lebanon', 'Uzbekistan', 'Palestinian Territories', 'Italy', 'Egypt', 'Albania', 'Netherlands', 'Mexico',
    'Azerbaijan', 'Norway', 'Sweden', 'Czech Republic', 'United Arab Emirates', 'Uganda', 'South Africa',
    'Honduras', 'Argentina', 'Belgium', 'Cyprus', 'Ecuador', 'Philippines', 'Puerto Rico', 'Holy See',
    'Greece', 'Brazil', 'Morocco', 'Estonia', 'Poland', 'Iceland', 'Indonesia', 'Kuwait', 'Ireland', 'Panama',
    'Jordan', 'Qatar', 'Tanzania', 'Turkey', 'Slovakia', 'Micronesia, Federated States of', 'Colombia',
    'Tunisia', 'Algeria', 'Malta', 'Nepal', 'Dominican Republic', 'Macao', 'Bosnia and Herzegovina',
    'Austria', 'Lithuania', 'Macedonia', 'Vietnam', 'South Korea', 'Romania', 'Cote d&#039;Ivoire', 'Reunion',
    'Sri Lanka', 'Chile', 'Armenia', 'Japan', 'Cayman Islands', 'Isle of Man', 'Rwanda', 'Gabon',
    'Saint Kitts and Nevis', 'Hungary', 'Kazakhstan', 'Zambia', 'Taiwan', 'New Caledonia', 'Barbados',
    'Slovenia', 'Moldova', 'Oman', 'Venezuela', 'Montenegro', 'Paraguay', 'Bolivia', 'French Polynesia',
    'Zimbabwe', 'Sint Maarten (Dutch part)', 'Trinidad and Tobago', 'Botswana', 'Ethiopia', 'Somalia',
    'Gibraltar', 'Antigua and Barbuda', 'Latvia', 'Ghana', 'Kyrgyzstan', 'Jamaica', 'Jersey', 'Russia',
    'Bermuda', 'Mali', 'Cameroon', 'Bahamas', 'Maldives', 'Benin', 'Mongolia', 'Guernsey',
    'Netherlands Antilles', 'Uruguay', 'Curacao', 'Malawi', 'Aland Islands', 'Mauritius', 'Cambodia',
    'United States Minor Outlying Islands', 'Seychelles', 'Guatemala', 'Namibia', 'Timor-Leste', 'Haiti',
    'Mozambique', 'Tajikistan', 'American Samoa', 'Andorra', 'El Salvador', 'British Virgin Islands',
    'Grenada', 'Sierra Leone', 'Mauritania', 'Yemen', 'Anguilla', 'Myanmar', 'Angola', 'Senegal',
    'Papua New Guinea', 'San Marino', 'Djibouti', 'Guyana', 'Togo', 'Belize', 'Comoros', 'Guinea',
    'Liechtenstein', 'Aruba', 'Saint Lucia', 'Guam', 'Madagascar', 'Nicaragua',
    'Saint Vincent and the Grenadines', 'Bhutan', 'Kiribati', 'Gambia', 'Swaziland', 'Saint Helena',
    'United States Virgin Islands', 'Turkmenistan', 'Belarus', 'Vanuatu', 'Brunei Darussalam',
    'Congo, the Democratic Republic of the', 'Dominica', 'Niue', 'Guadeloupe',
    'Turks and Caicos Islands', 'Tuvalu', 'Laos', 'Monaco', 'Fiji', 'Martinique',
    'Bonaire, Sint Eustatius and Saba', 'Congo', 'Suriname', 'Central African Republic', 'Faroe Islands',
    'Chad', 'Northern Mariana Islands', 'Palau', 'Eritrea', 'Burkina Faso', 'Samoa',
    'Cocos (Keeling) Islands', 'Niger', 'Burundi', 'Cook Islands', 'Greenland', 'French Guiana'
])

# Example company names
companies = [
    "Google", "Microsoft", "Amazon", "Meta", "Apple", "Netflix",
    "Adobe", "Oracle", "Salesforce", "SAP"
]

# Get categories from dataset
categories = sorted(jobs_df['category'].dropna().unique().tolist())

# Sidebar and Title
st.set_page_config(page_title="Job Recommender", layout="wide")
st.title("🏋️ Job Recommendation Engine")

# --- Form ---
with st.form(key="recommend_form"):
    st.markdown("### 📘 Job Description")
    job_desc = st.text_area("Describe the job role you're looking for", height=150)

    st.markdown("### 📂 Select Category (Optional)")
    job_category = st.selectbox("Category", [""] + categories)

    st.markdown("### 🏢 Company (Type or select)")
    company_input = st.text_input("Enter Company Name (or select below)")
    job_company = st.selectbox("Or pick from list", [""] + companies)
    job_company = company_input if company_input else job_company

    st.markdown("### 🌍 Location (Country)")
    job_location = st.selectbox("Select a country", [""] + locations)

    submit = st.form_submit_button("🔎 Recommend Jobs")

# --- Recommendation Logic ---
if submit:
    if not job_desc:
        st.warning("Please provide a job description.")
    else:
        # Prepare input text
        input_text = job_desc + " " + job_category + " " + job_company + " " + job_location

        # Vectorize input
        input_vec = tfidf_vectorizer.transform([input_text])
        job_vecs = tfidf_vectorizer.transform(jobs_df['processed_text'])

        # Compute cosine similarity
        sims = cosine_similarity(input_vec, job_vecs).flatten()
        jobs_df['similarity'] = sims
        top_jobs = jobs_df.sort_values(by='similarity', ascending=False).head(10)

        # Display results
        st.markdown("## 🔍 Top Recommended Jobs")
        for _, row in top_jobs.iterrows():
            st.markdown(f"**{row['title']}**")
            st.markdown(f"- 🏢 **Company:** {job_company if job_company else 'N/A'}")
            st.markdown(f"- 🌍 **Location:** {row['country']}")
            st.markdown(f"- ✔️ **Category:** {row['category']}")
            st.markdown(f"- ➜ [View Job Posting]({row['link']})")
            st.markdown("---")

        # Optional download
        st.markdown("### 📂 Download Results")
        download_cols = ['title', 'link', 'country', 'category', 'similarity']
        csv_data = top_jobs[download_cols].to_csv(index=False)
        st.download_button("📄 Download as CSV", csv_data, file_name="recommended_jobs.csv", mime="text/csv")
