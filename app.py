import streamlit as st
import os
import json
import requests
import tempfile
from pypdf import PdfReader
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="CareerFit AI",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 CareerFit AI")
st.markdown("*Upload your resume + paste a job description → get your match score, skill gaps, and rewritten bullets*")
st.divider()

# ── Sidebar: API keys ─────────────────────────────────────────
with st.sidebar:
    st.header("🔑 API Keys")
    groq_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...")
    rapidapi_key = st.text_input("RapidAPI Key", type="password", placeholder="your rapidapi key")
    st.divider()
    st.header("🔍 Job Search Settings")
    job_title = st.text_input("Target Job Title", placeholder="e.g. Data Scientist")
    job_location = st.text_input("Location", placeholder="e.g. New York or Remote")
    fetch_jobs = st.button("🌐 Fetch Live Job Postings", use_container_width=True)

# ── Helper functions ──────────────────────────────────────────
def extract_pdf_text(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(uploaded_file.read())
        tmp_path = f.name
    reader = PdfReader(tmp_path)
    return "\n".join(page.extract_text() for page in reader.pages if page.extract_text())

def fetch_live_jobs(job_title, location, rapidapi_key, num_results=10):
    url = "https://jsearch.p.rapidapi.com/search"
    headers = {
        "X-RapidAPI-Key": rapidapi_key,
        "X-RapidAPI-Host": "jsearch.p.rapidapi.com"
    }
    params = {
        "query": f"{job_title} in {location}",
        "num_pages": "1",
        "page": "1"
    }
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        data = response.json()
        jobs = data.get("data", [])[:num_results]
        return jobs
    return []

def build_vectorstore(text, collection_name="resume"):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        collection_name=collection_name
    ) 
    return vectorstore

def analyze_fit(resume_text, jd_text, groq_key):
    llm = ChatGroq(
        groq_api_key=groq_key,
        model_name="openai/gpt-oss-120b",
        temperature=0.3
    )

    prompt = f"""
You are an expert career coach and recruiter. Analyze the resume against the job description below.

RESUME:
{resume_text[:3000]}

JOB DESCRIPTION:
{jd_text[:3000]}

Provide a structured analysis in the following JSON format ONLY, no extra text:
{{
    "overall_score": <integer 0-100>,
    "technical_skills_score": <integer 0-100>,
    "experience_score": <integer 0-100>,
    "education_score": <integer 0-100>,
    "matched_skills": [<list of matching skills>],
    "missing_skills": [<list of missing skills>],
    "strengths": [<list of 3 strengths>],
    "weaknesses": [<list of 3 weaknesses>],
    "rewritten_bullets": [<list of 3 rewritten resume bullets tailored to JD>],
    "verdict": "<one sentence summary>"
}}
"""
    response = llm.invoke(prompt)
    raw = response.content.strip()
    # Clean up response
    if "```json" in raw:
        raw = raw.split("```json")[1].split("```")[0].strip()
    elif "```" in raw:
        raw = raw.split("```")[1].split("```")[0].strip()
    return json.loads(raw)

def get_learning_roadmap(missing_skills, groq_key):
    llm = ChatGroq(
        groq_api_key=groq_key,
        model_name="openai/gpt-oss-120b",
        temperature=0.1
    )
    skills_str = ", ".join(missing_skills[:5])
    prompt = f"""
For each of these missing skills: {skills_str}

Provide a learning roadmap in JSON format ONLY:
{{
    "roadmap": [
        {{
            "skill": "<skill name>",
            "time_to_learn": "<e.g. 2 weeks>",
            "free_resource": "<specific course or resource URL>",
            "resume_line": "<how to add this to resume once learned>"
        }}
    ]
}}
"""
    response = llm.invoke(prompt)
    raw = response.content.strip()
    if "```json" in raw:
        raw = raw.split("```json")[1].split("```")[0].strip()
    elif "```" in raw:
        raw = raw.split("```")[1].split("```")[0].strip()
    return json.loads(raw)

# ── Main UI ───────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("📄 Your Resume")
    resume_file = st.file_uploader("Upload Resume PDF", type="pdf")

with col2:
    st.subheader("📋 Job Description")
    jd_input_method = st.radio("Input method", ["Paste text", "Upload PDF"], horizontal=True)
    if jd_input_method == "Paste text":
        jd_text_input = st.text_area("Paste job description here", height=200)
    else:
        jd_file = st.file_uploader("Upload JD PDF", type="pdf")

# ── Fetch live jobs ───────────────────────────────────────────
if fetch_jobs:
    if not rapidapi_key or not job_title:
        st.sidebar.error("Enter your RapidAPI key and job title first!")
    else:
        with st.spinner("🔍 Fetching live job postings from Dice & LinkedIn..."):
            jobs = fetch_live_jobs(job_title, job_location or "Remote", rapidapi_key)
            if jobs:
                st.subheader(f"🌐 Live Job Postings for '{job_title}'")
                for i, job in enumerate(jobs):
                    with st.expander(f"#{i+1} {job.get('job_title', 'N/A')} @ {job.get('employer_name', 'N/A')}"):
                        st.write(f"📍 **Location:** {job.get('job_city', 'Remote')}")
                        st.write(f"🔗 **Apply:** {job.get('job_apply_link', 'N/A')}")
                        desc = job.get('job_description', '')[:500]
                        st.write(f"📝 **Description:** {desc}...")
            else:
                st.sidebar.error("No jobs found. Check your API key or try different search terms.")

st.divider()

# ── Analyze button ────────────────────────────────────────────
analyze_btn = st.button("🚀 Analyze My Fit", use_container_width=True, type="primary")

if analyze_btn:
    if not groq_key:
        st.error("Please enter your Groq API key in the sidebar!")
    elif not resume_file:
        st.error("Please upload your resume PDF!")
    else:
        # Extract texts
        resume_text = extract_pdf_text(resume_file)

        if jd_input_method == "Paste text":
            jd_text = jd_text_input
        else:
            if 'jd_file' in locals() and jd_file:
                jd_text = extract_pdf_text(jd_file)
            else:
                st.error("Please upload or paste a job description!")
                st.stop()

        if not jd_text.strip():
            st.error("Job description is empty!")
            st.stop()

        with st.spinner("🤖 Analyzing your fit... this takes ~15 seconds"):
            try:
                result = analyze_fit(resume_text, jd_text, groq_key)

                # ── Scores ────────────────────────────────────────────
                st.subheader("📊 Match Analysis")
                col_a, col_b, col_c, col_d = st.columns(4)
                col_a.metric("🎯 Overall Fit", f"{result['overall_score']}%")
                col_b.metric("💻 Technical Skills", f"{result['technical_skills_score']}%")
                col_c.metric("💼 Experience", f"{result['experience_score']}%")
                col_d.metric("🎓 Education", f"{result['education_score']}%")

                # Score bar
                score = result['overall_score']
                if score >= 75:
                    st.success(f"✅ Strong Match! {result['verdict']}")
                elif score >= 50:
                    st.warning(f"⚠️ Moderate Match. {result['verdict']}")
                else:
                    st.error(f"❌ Weak Match. {result['verdict']}")

                st.progress(score / 100)
                st.divider()

                # ── Skills ────────────────────────────────────────────
                col_e, col_f = st.columns(2)
                with col_e:
                    st.subheader("✅ Matched Skills")
                    for skill in result.get('matched_skills', []):
                        st.success(f"✓ {skill}")

                with col_f:
                    st.subheader("⚠️ Missing Skills")
                    for skill in result.get('missing_skills', []):
                        st.error(f"✗ {skill}")

                st.divider()

                # ── Strengths & Weaknesses ────────────────────────────
                col_g, col_h = st.columns(2)
                with col_g:
                    st.subheader("💪 Strengths")
                    for s in result.get('strengths', []):
                        st.write(f"• {s}")

                with col_h:
                    st.subheader("🔧 Areas to Improve")
                    for w in result.get('weaknesses', []):
                        st.write(f"• {w}")

                st.divider()

                # ── Rewritten bullets ─────────────────────────────────
                st.subheader("✏️ AI-Rewritten Resume Bullets (Tailored to This JD)")
                st.caption("Copy these into your resume to improve keyword match")
                for bullet in result.get('rewritten_bullets', []):
                    st.info(f"• {bullet}")

                st.divider()

                # ── Learning roadmap ──────────────────────────────────
                missing = result.get('missing_skills', [])
                if missing:
                    st.subheader("🗺️ Your Learning Roadmap")
                    st.caption("How to close your skill gaps")
                    with st.spinner("Building your personalized roadmap..."):
                        roadmap = get_learning_roadmap(missing, groq_key)
                        for item in roadmap.get('roadmap', []):
                            with st.expander(f"📚 {item['skill']} — {item['time_to_learn']}"):
                                st.write(f"🔗 **Resource:** {item['free_resource']}")
                                st.write(f"📝 **Add to resume as:** {item['resume_line']}")

                # ── Save to history ───────────────────────────────────
                if 'history' not in st.session_state:
                    st.session_state.history = []

                st.session_state.history.append({
                    "score": result['overall_score'],
                    "verdict": result['verdict'],
                    "missing": missing
                })

            except Exception as e:
                st.error(f"Something went wrong: {str(e)}")
                st.info("Make sure your Groq API key is valid and try again.")

# ── History ───────────────────────────────────────────────────
if 'history' in st.session_state and st.session_state.history:
    st.divider()
    st.subheader("📁 Application History (This Session)")
    for i, h in enumerate(st.session_state.history):
        st.write(f"**Application #{i+1}** — Score: {h['score']}% | {h['verdict']}")
